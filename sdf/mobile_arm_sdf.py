from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import jax.numpy as jnp
from jax import lax

from sdf.geometry import ObstacleField
from sdf.nsdf_network import (
    legacy_sdf_values,
    legacy_sdf_values_and_gradients,
    params_to_jax,
    validate_legacy_params,
)


NSDF_DTYPE = jnp.float32
MOBILE_ARM_STATE_DIM = 11
MOBILE_ARM_PART_COUNT = 9

_BASE_LENGTH = 1.5
_BASE_WIDTH = 1.0
_ARM_OFFSETS = ((-0.5, 0.0), (0.5, 0.0))
_LINK_LENGTHS = (0.8, 0.6, 0.4, 0.2)
_CANONICAL_LINK_LENGTH = 0.8
_CANONICAL_LINK_WIDTH = 0.2
_PRODUCTION_MODE = "narrow_surface"
_DIAGNOSTIC_MODES = ("circle_center", "surface_points")


@dataclass
class MobileArmSDF:
    """Compositional neural SDF for the repository's articulated mobile arm.

    ``base_params`` must represent the base's 1.5-by-1.0 rectangle and
    ``link_params`` must represent the canonical 0.8-by-0.2 link. All eight
    links have the same 4:1 aspect ratio, so their signed distances are
    obtained by uniformly scaling the canonical model.

    The production ``narrow_surface`` mode uses exact rectangle geometry only
    to choose a fixed number of nearest rigid parts and guard learned
    overestimation. It then evaluates a fixed obstacle-surface stencil against
    those parts. Queries outside each primitive's training domain are clamped
    before network evaluation and masked from the learned minimum.

    ``circle_center`` and ``surface_points`` retain the original all-part
    implementations for explicit diagnostics. They can extrapolate outside
    the learned domain and are not production defaults.
    """

    base_params: Mapping[str, Any]
    link_params: Mapping[str, Any]
    base_metadata: Mapping[str, Any] | None = None
    link_metadata: Mapping[str, Any] | None = None
    base_description: str = "mobile-arm base 1.5 x 1.0"
    link_description: str = "mobile-arm canonical link 0.8 x 0.2"
    geometry_padding: float = 0.02
    model_margin: float = 0.0
    base_model_margin: float = 0.0
    link_model_margin: float = 0.0
    base_domain_padding: float | None = None
    link_domain_padding: float | None = None
    obstacle_mode: Literal[
        "narrow_surface",
        "circle_center",
        "surface_points",
    ] = _PRODUCTION_MODE
    points_per_obstacle: int = 64
    narrow_phase_points: int = 4
    broad_phase_parts: int = 2
    analytic_guard: bool = True
    learned_blend_full_distance: float = 0.05
    learned_blend_zero_distance: float = 0.20

    def __post_init__(self) -> None:
        if self.geometry_padding < 0.0:
            raise ValueError("geometry_padding must be nonnegative")
        for name, margin in (
            ("model_margin", self.model_margin),
            ("base_model_margin", self.base_model_margin),
            ("link_model_margin", self.link_model_margin),
        ):
            if margin < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.obstacle_mode not in (_PRODUCTION_MODE, *_DIAGNOSTIC_MODES):
            raise ValueError(
                "obstacle_mode must be 'narrow_surface', 'circle_center', "
                "or 'surface_points'"
            )
        if self.points_per_obstacle < 4:
            raise ValueError("points_per_obstacle must be at least 4")
        if not 1 <= self.narrow_phase_points <= self.points_per_obstacle:
            raise ValueError(
                "narrow_phase_points must be between 1 and "
                "points_per_obstacle"
            )
        if not 1 <= self.broad_phase_parts <= MOBILE_ARM_PART_COUNT:
            raise ValueError(
                f"broad_phase_parts must be between 1 and "
                f"{MOBILE_ARM_PART_COUNT}"
            )
        if self.learned_blend_full_distance < 0.0:
            raise ValueError(
                "learned_blend_full_distance must be nonnegative"
            )
        if (
            self.learned_blend_zero_distance
            <= self.learned_blend_full_distance
        ):
            raise ValueError(
                "learned_blend_zero_distance must be greater than "
                "learned_blend_full_distance"
            )

        self.base_params = params_to_jax(self.base_params)
        self.link_params = params_to_jax(self.link_params)
        validate_legacy_params(self.base_params)
        validate_legacy_params(self.link_params)
        _validate_rectangle_metadata(
            self.base_metadata,
            expected_width=_BASE_LENGTH,
            expected_height=_BASE_WIDTH,
            label="base",
        )
        _validate_rectangle_metadata(
            self.link_metadata,
            expected_width=_CANONICAL_LINK_LENGTH,
            expected_height=_CANONICAL_LINK_WIDTH,
            label="canonical link",
        )
        self.base_domain_padding = _resolve_domain_padding(
            self.base_domain_padding,
            self.base_metadata,
            label="base",
            required=self.obstacle_mode == _PRODUCTION_MODE,
        )
        self.link_domain_padding = _resolve_domain_padding(
            self.link_domain_padding,
            self.link_metadata,
            label="canonical link",
            required=self.obstacle_mode == _PRODUCTION_MODE,
        )

    @property
    def description(self) -> str:
        margin = self.geometry_padding + self.model_margin
        return (
            f"compositional mobile-arm NSDF "
            f"(base={self.base_description}; link={self.link_description}; "
            f"mode={self.obstacle_mode}; total_margin={margin:g}; "
            f"broad_parts={self.broad_phase_parts}; "
            f"narrow_points={self.narrow_phase_points}; "
            f"analytic_guard={self.analytic_guard})"
        )

    def obstacle_barriers(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> jnp.ndarray:
        """Return one conservative neural barrier value per circle obstacle."""
        state = _validate_state(state)
        if not field.obstacles:
            return jnp.zeros((0,), dtype=NSDF_DTYPE)
        if self.obstacle_mode == _PRODUCTION_MODE:
            context = self._narrow_phase_context(state, field)
            return context["values"]
        query_points, radius_offsets = self._obstacle_queries(field)
        candidate_values, _local_points = self._candidate_values(state, query_points)
        adjusted = candidate_values - radius_offsets[:, :, None] - self._total_margin
        flattened = adjusted.reshape((adjusted.shape[0], -1))
        nearest = jnp.argmin(flattened, axis=1)
        return flattened[jnp.arange(flattened.shape[0]), nearest]

    def obstacle_barriers_and_jacobian(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return barriers and their direct full 11-state Jacobian.

        Values are first evaluated for all obstacle-query/robot-part pairs.
        Local network gradients are then evaluated only at the winning query
        and part for each obstacle and chained through the rigid-part pose
        Jacobians. At ties, the gradient follows ``jnp.argmin``'s first-index
        subgradient, consistent with the value path.
        """
        state = _validate_state(state)
        if not field.obstacles:
            return (
                jnp.zeros((0,), dtype=NSDF_DTYPE),
                jnp.zeros((0, MOBILE_ARM_STATE_DIM), dtype=NSDF_DTYPE),
            )
        if self.obstacle_mode == _PRODUCTION_MODE:
            return self._narrow_phase_values_and_jacobian(state, field)
        query_points, radius_offsets = self._obstacle_queries(field)
        candidate_values, local_points = self._candidate_values(state, query_points)
        adjusted = candidate_values - radius_offsets[:, :, None] - self._total_margin

        obstacle_count, query_count, _part_count = adjusted.shape
        flattened = adjusted.reshape((obstacle_count, -1))
        nearest = jnp.argmin(flattened, axis=1)
        row = jnp.arange(obstacle_count)
        values = flattened[row, nearest]

        query_index = nearest // MOBILE_ARM_PART_COUNT
        part_index = nearest % MOBILE_ARM_PART_COUNT
        selected_world_points = query_points[row, query_index]
        selected_local_points = local_points[row, query_index, part_index]
        scales = _part_scales()
        selected_scales = scales[part_index]
        normalized_points = selected_local_points / selected_scales[:, None]

        # Dynamic grouping would make the compiled shape depend on which part
        # wins. Evaluating both small primitive networks for O selected points
        # remains fixed-shape and is substantially cheaper than differentiating
        # all 9*O candidate evaluations.
        _base_values, base_gradients = legacy_sdf_values_and_gradients(
            self.base_params,
            normalized_points,
        )
        _link_values, link_gradients = legacy_sdf_values_and_gradients(
            self.link_params,
            normalized_points,
        )
        is_base = part_index == 0
        local_gradients = jnp.where(
            is_base[:, None],
            base_gradients[:, :2],
            link_gradients[:, :2],
        )

        (
            part_centers,
            part_angles,
            part_center_jacobians,
            part_angle_jacobians,
        ) = _part_poses_and_jacobians(state)
        selected_centers = part_centers[part_index]
        selected_angles = part_angles[part_index]
        selected_center_jacobians = part_center_jacobians[part_index]
        selected_angle_jacobians = part_angle_jacobians[part_index]
        selected_local_jacobians = _world_to_local_jacobians(
            selected_world_points,
            selected_centers,
            selected_angles,
            selected_center_jacobians,
            selected_angle_jacobians,
        )
        jacobian = jnp.einsum(
            "oi,ois->os",
            local_gradients,
            selected_local_jacobians,
        )
        return values, jacobian

    def _narrow_phase_context(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> dict[str, jnp.ndarray]:
        """Select exact-nearest parts, then query only those learned SDFs."""
        obstacle_centers = jnp.asarray(field.centers, dtype=NSDF_DTYPE)
        obstacle_radii = jnp.asarray(field.radii, dtype=NSDF_DTYPE)
        part_centers, part_angles = _part_poses(state)
        center_local_points = _world_points_to_all_part_frames(
            obstacle_centers,
            part_centers,
            part_angles,
        )
        part_half_extents = _part_half_extents()
        exact_part_distances, exact_part_local_gradients = (
            _rectangle_values_and_gradients(
                center_local_points,
                part_half_extents[None, :, :],
            )
        )
        analytic_part_index = jnp.argmin(exact_part_distances, axis=1)
        row = jnp.arange(obstacle_centers.shape[0])
        exact_distance = exact_part_distances[row, analytic_part_index]
        exact_local_gradient = exact_part_local_gradients[
            row,
            analytic_part_index,
        ]
        analytic_values = (
            exact_distance
            - obstacle_radii
            - jnp.asarray(self.geometry_padding, dtype=NSDF_DTYPE)
        )

        _negative_part_distances, broad_part_indices = lax.top_k(
            -exact_part_distances,
            self.broad_phase_parts,
        )
        surface_point_bank = self._surface_points(field)
        broad_centers = part_centers[broad_part_indices]
        broad_angles = part_angles[broad_part_indices]
        broad_scales = _part_scales()[broad_part_indices]
        surface_bank_local_points = _world_points_to_selected_parts_frames(
            surface_point_bank,
            broad_centers,
            broad_angles,
        )
        selected_half_extents = part_half_extents[broad_part_indices]
        surface_bank_exact_distances, _surface_bank_gradients = (
            _rectangle_values_and_gradients(
                surface_bank_local_points,
                selected_half_extents[:, :, None, :],
            )
        )
        _negative_distances, narrow_indices = lax.top_k(
            -surface_bank_exact_distances,
            self.narrow_phase_points,
        )
        obstacle_count = obstacle_centers.shape[0]
        surface_bank_broadcast = jnp.broadcast_to(
            surface_point_bank[:, None, :, :],
            (
                obstacle_count,
                self.broad_phase_parts,
                self.points_per_obstacle,
                2,
            ),
        )
        gather_index = narrow_indices[:, :, :, None]
        surface_points = jnp.take_along_axis(
            surface_bank_broadcast,
            gather_index,
            axis=2,
        )
        surface_local_points = jnp.take_along_axis(
            surface_bank_local_points,
            gather_index,
            axis=2,
        )
        normalized_points = (
            surface_local_points / broad_scales[:, :, None, None]
        )
        domain_half_extents = self._selected_domain_half_extents(
            broad_part_indices
        )
        in_domain = jnp.all(
            jnp.abs(normalized_points)
            < domain_half_extents[:, :, None, :],
            axis=3,
        )
        guarded_points = jnp.clip(
            normalized_points,
            -domain_half_extents[:, :, None, :],
            domain_half_extents[:, :, None, :],
        )
        base_domain, link_domain = self._primitive_domain_half_extents()
        base_guarded_points = jnp.clip(
            normalized_points,
            -base_domain[None, None, None, :],
            base_domain[None, None, None, :],
        )
        link_guarded_points = jnp.clip(
            normalized_points,
            -link_domain[None, None, None, :],
            link_domain[None, None, None, :],
        )

        point_count = (
            obstacle_count
            * self.broad_phase_parts
            * self.narrow_phase_points
        )
        base_predictions = legacy_sdf_values(
            self.base_params,
            base_guarded_points.reshape((point_count, 2)),
        ).reshape(
            (
                obstacle_count,
                self.broad_phase_parts,
                self.narrow_phase_points,
            )
        )
        link_predictions = legacy_sdf_values(
            self.link_params,
            link_guarded_points.reshape((point_count, 2)),
        ).reshape(
            (
                obstacle_count,
                self.broad_phase_parts,
                self.narrow_phase_points,
            )
        )
        primitive_predictions = jnp.where(
            (broad_part_indices == 0)[:, :, None],
            base_predictions,
            link_predictions,
        )
        physical_predictions = (
            primitive_predictions * broad_scales[:, :, None]
        )
        learned_margins = self._selected_model_margins(
            broad_part_indices,
            broad_scales,
        )
        learned_candidates = (
            physical_predictions
            - jnp.asarray(self.geometry_padding, dtype=NSDF_DTYPE)
            - learned_margins[:, :, None]
        )
        learned_candidates = jnp.where(
            in_domain & jnp.isfinite(learned_candidates),
            learned_candidates,
            jnp.inf,
        )
        flattened_candidates = learned_candidates.reshape(
            (obstacle_count, -1)
        )
        learned_index = jnp.argmin(flattened_candidates, axis=1)
        broad_slot = learned_index // self.narrow_phase_points
        point_index = learned_index % self.narrow_phase_points
        learned_values = flattened_candidates[row, learned_index]
        has_learned_value = jnp.any(
            jnp.isfinite(flattened_candidates),
            axis=1,
        )
        part_index = broad_part_indices[row, broad_slot]
        if self.analytic_guard:
            use_learned = has_learned_value & (
                learned_values < analytic_values
            )
            corrected_values = jnp.where(
                use_learned,
                learned_values,
                analytic_values,
            )
            blend_weight, blend_derivative = _guard_blend_weight(
                analytic_values,
                full_distance=self.learned_blend_full_distance,
                zero_distance=self.learned_blend_zero_distance,
            )
            values = (
                analytic_values
                + blend_weight * (corrected_values - analytic_values)
            )
        else:
            use_learned = has_learned_value
            corrected_values = jnp.where(
                use_learned,
                learned_values,
                analytic_values,
            )
            blend_weight = jnp.where(
                use_learned,
                1.0,
                0.0,
            ).astype(NSDF_DTYPE)
            blend_derivative = jnp.zeros_like(blend_weight)
            values = corrected_values
        return {
            "values": values,
            "analytic_values": analytic_values,
            "corrected_values": corrected_values,
            "learned_values": learned_values,
            "exact_local_gradient": exact_local_gradient,
            "analytic_part_index": analytic_part_index,
            "part_index": part_index,
            "broad_slot": broad_slot,
            "point_index": point_index,
            "broad_part_indices": broad_part_indices,
            "surface_points": surface_points,
            "surface_local_points": surface_local_points,
            "normalized_points": normalized_points,
            "guarded_points": guarded_points,
            "use_learned": use_learned,
            "blend_weight": blend_weight,
            "blend_derivative": blend_derivative,
        }

    def _narrow_phase_values_and_jacobian(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Differentiate the guarded narrow phase through the selected part."""
        context = self._narrow_phase_context(state, field)
        obstacle_count = len(field.obstacles)
        row = jnp.arange(obstacle_count)
        part_index = context["part_index"]
        broad_slot = context["broad_slot"]
        point_index = context["point_index"]

        selected_normalized_points = context["normalized_points"][
            row,
            broad_slot,
            point_index,
        ]
        base_domain, link_domain = self._primitive_domain_half_extents()
        _base_values, base_gradients = legacy_sdf_values_and_gradients(
            self.base_params,
            jnp.clip(
                selected_normalized_points,
                -base_domain[None, :],
                base_domain[None, :],
            ),
        )
        _link_values, link_gradients = legacy_sdf_values_and_gradients(
            self.link_params,
            jnp.clip(
                selected_normalized_points,
                -link_domain[None, :],
                link_domain[None, :],
            ),
        )
        learned_local_gradients = jnp.where(
            (part_index == 0)[:, None],
            base_gradients[:, :2],
            link_gradients[:, :2],
        )

        (
            part_centers,
            part_angles,
            part_center_jacobians,
            part_angle_jacobians,
        ) = _part_poses_and_jacobians(state)
        selected_centers = part_centers[part_index]
        selected_angles = part_angles[part_index]
        selected_center_jacobians = part_center_jacobians[part_index]
        selected_angle_jacobians = part_angle_jacobians[part_index]

        selected_surface_points = context["surface_points"][
            row,
            broad_slot,
            point_index,
        ]
        learned_local_jacobians = _world_to_local_jacobians(
            selected_surface_points,
            selected_centers,
            selected_angles,
            selected_center_jacobians,
            selected_angle_jacobians,
        )
        learned_jacobian = jnp.einsum(
            "oi,ois->os",
            learned_local_gradients,
            learned_local_jacobians,
        )

        obstacle_centers = jnp.asarray(field.centers, dtype=NSDF_DTYPE)
        analytic_part_index = context["analytic_part_index"]
        analytic_centers = part_centers[analytic_part_index]
        analytic_angles = part_angles[analytic_part_index]
        analytic_center_jacobians = part_center_jacobians[
            analytic_part_index
        ]
        analytic_angle_jacobians = part_angle_jacobians[
            analytic_part_index
        ]
        analytic_local_jacobians = _world_to_local_jacobians(
            obstacle_centers,
            analytic_centers,
            analytic_angles,
            analytic_center_jacobians,
            analytic_angle_jacobians,
        )
        analytic_jacobian = jnp.einsum(
            "oi,ois->os",
            context["exact_local_gradient"],
            analytic_local_jacobians,
        )
        corrected_jacobian = jnp.where(
            context["use_learned"][:, None],
            learned_jacobian,
            analytic_jacobian,
        )
        blend_weight = context["blend_weight"][:, None]
        blend_derivative = context["blend_derivative"][:, None]
        correction = (
            context["corrected_values"] - context["analytic_values"]
        )[:, None]
        jacobian = (
            analytic_jacobian
            + blend_weight * (corrected_jacobian - analytic_jacobian)
            + correction * blend_derivative * analytic_jacobian
        )
        return context["values"], jacobian

    def _selected_domain_half_extents(
        self,
        part_index: jnp.ndarray,
    ) -> jnp.ndarray:
        base_domain, link_domain = self._primitive_domain_half_extents()
        return jnp.where(
            (part_index == 0)[..., None],
            base_domain,
            link_domain,
        )

    def _primitive_domain_half_extents(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        assert self.base_domain_padding is not None
        assert self.link_domain_padding is not None
        base_domain = jnp.asarray(
            (
                0.5 * _BASE_LENGTH + self.base_domain_padding,
                0.5 * _BASE_WIDTH + self.base_domain_padding,
            ),
            dtype=NSDF_DTYPE,
        )
        link_domain = jnp.asarray(
            (
                0.5 * _CANONICAL_LINK_LENGTH + self.link_domain_padding,
                0.5 * _CANONICAL_LINK_WIDTH + self.link_domain_padding,
            ),
            dtype=NSDF_DTYPE,
        )
        return base_domain, link_domain

    def _selected_model_margins(
        self,
        part_index: jnp.ndarray,
        selected_scales: jnp.ndarray,
    ) -> jnp.ndarray:
        base_margin = jnp.asarray(
            self.model_margin + self.base_model_margin,
            dtype=NSDF_DTYPE,
        )
        link_margin = (
            jnp.asarray(self.model_margin, dtype=NSDF_DTYPE)
            + selected_scales
            * jnp.asarray(self.link_model_margin, dtype=NSDF_DTYPE)
        )
        return jnp.where(part_index == 0, base_margin, link_margin)

    def _surface_points(self, field: ObstacleField) -> jnp.ndarray:
        centers = jnp.asarray(field.centers, dtype=NSDF_DTYPE)
        radii = jnp.asarray(field.radii, dtype=NSDF_DTYPE)
        theta = jnp.linspace(
            0.0,
            2.0 * jnp.pi,
            self.points_per_obstacle,
            endpoint=False,
            dtype=NSDF_DTYPE,
        )
        unit_circle = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=1)
        return (
            centers[:, None, :]
            + radii[:, None, None] * unit_circle[None, :, :]
        )

    @property
    def _total_margin(self) -> jnp.ndarray:
        return jnp.asarray(
            self.geometry_padding + self.model_margin,
            dtype=NSDF_DTYPE,
        )

    def _obstacle_queries(
        self,
        field: ObstacleField,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        obstacle_count = len(field.obstacles)
        if obstacle_count == 0:
            return (
                jnp.zeros((0, 1, 2), dtype=NSDF_DTYPE),
                jnp.zeros((0, 1), dtype=NSDF_DTYPE),
            )

        centers = jnp.asarray(field.centers, dtype=NSDF_DTYPE)
        radii = jnp.asarray(field.radii, dtype=NSDF_DTYPE)
        if self.obstacle_mode == "circle_center":
            return centers[:, None, :], radii[:, None]

        theta = jnp.linspace(
            0.0,
            2.0 * jnp.pi,
            self.points_per_obstacle,
            endpoint=False,
            dtype=NSDF_DTYPE,
        )
        unit_circle = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=1)
        surface_points = (
            centers[:, None, :]
            + radii[:, None, None] * unit_circle[None, :, :]
        )
        return surface_points, jnp.zeros(
            (obstacle_count, self.points_per_obstacle),
            dtype=NSDF_DTYPE,
        )

    def _candidate_values(
        self,
        state: jnp.ndarray,
        query_points: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return raw part SDF values and physical local coordinates."""
        centers, angles = _part_poses(state)
        delta = query_points[:, :, None, :] - centers[None, None, :, :]
        c = jnp.cos(angles)
        s = jnp.sin(angles)
        local_x = delta[..., 0] * c[None, None, :] + delta[..., 1] * s[
            None, None, :
        ]
        local_y = -delta[..., 0] * s[None, None, :] + delta[..., 1] * c[
            None, None, :
        ]
        local_points = jnp.stack((local_x, local_y), axis=-1)

        obstacle_count, query_count = query_points.shape[:2]
        flat_count = obstacle_count * query_count
        scales = _part_scales()
        normalized = local_points / scales[None, None, :, None]
        base_points = normalized[:, :, 0, :].reshape((flat_count, 2))
        link_points = normalized[:, :, 1:, :].reshape(
            (flat_count * (MOBILE_ARM_PART_COUNT - 1), 2)
        )
        base_values = legacy_sdf_values(self.base_params, base_points).reshape(
            (obstacle_count, query_count, 1)
        )
        link_values = legacy_sdf_values(self.link_params, link_points).reshape(
            (obstacle_count, query_count, MOBILE_ARM_PART_COUNT - 1)
        )
        link_values = link_values * scales[None, None, 1:]
        values = jnp.concatenate((base_values, link_values), axis=2)
        return values, local_points


def _validate_state(state: jnp.ndarray) -> jnp.ndarray:
    state = jnp.asarray(state, dtype=NSDF_DTYPE)
    if state.ndim != 1 or state.shape[0] != MOBILE_ARM_STATE_DIM:
        raise ValueError(
            f"mobile-arm NSDF state must have shape ({MOBILE_ARM_STATE_DIM},), "
            f"got {state.shape}"
        )
    return state


def _validate_rectangle_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    expected_width: float,
    expected_height: float,
    label: str,
) -> None:
    """Reject accidentally swapped or unrelated primitive checkpoints."""
    if metadata is None:
        return
    geometry = metadata.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"{label} metadata is missing a geometry mapping")
    if geometry.get("type") != "axis_aligned_rectangle":
        raise ValueError(
            f"{label} metadata geometry.type must be "
            "'axis_aligned_rectangle'"
        )
    try:
        width = float(geometry["width"])
        height = float(geometry["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} metadata geometry must contain numeric width and height"
        ) from exc
    tolerance = 1e-6
    if (
        abs(width - expected_width) > tolerance
        or abs(height - expected_height) > tolerance
    ):
        raise ValueError(
            f"{label} checkpoint geometry must be "
            f"{expected_width:g} x {expected_height:g}, "
            f"got {width:g} x {height:g}"
        )


def _resolve_domain_padding(
    explicit_padding: float | None,
    metadata: Mapping[str, Any] | None,
    *,
    label: str,
    required: bool,
) -> float:
    if explicit_padding is not None:
        padding = float(explicit_padding)
    else:
        training = metadata.get("training") if metadata is not None else None
        config = training.get("config") if isinstance(training, Mapping) else None
        stored_padding = (
            config.get("domain_padding")
            if isinstance(config, Mapping)
            else None
        )
        if stored_padding is None and required:
            raise ValueError(
                f"{label} production NSDF requires either explicit "
                "domain padding or training metadata containing "
                "training.config.domain_padding"
            )
        padding = 1.0 if stored_padding is None else float(stored_padding)
    if not math.isfinite(padding) or padding <= 0.0:
        raise ValueError(f"{label} domain padding must be finite and positive")
    return padding


def _part_scales() -> jnp.ndarray:
    return jnp.asarray(
        (
            1.0,
            *(
                length / _CANONICAL_LINK_LENGTH
                for _arm in range(2)
                for length in _LINK_LENGTHS
            ),
        ),
        dtype=NSDF_DTYPE,
    )


def _part_half_extents() -> jnp.ndarray:
    scales = _part_scales()
    base_half_extents = jnp.asarray(
        ((0.5 * _BASE_LENGTH, 0.5 * _BASE_WIDTH),),
        dtype=NSDF_DTYPE,
    )
    link_half_extents = scales[1:, None] * jnp.asarray(
        (0.5 * _CANONICAL_LINK_LENGTH, 0.5 * _CANONICAL_LINK_WIDTH),
        dtype=NSDF_DTYPE,
    )[None, :]
    return jnp.concatenate((base_half_extents, link_half_extents), axis=0)


def _rectangle_values_and_gradients(
    points: jnp.ndarray,
    half_extents: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact rectangle SDF and a deterministic local-frame subgradient."""
    offset = jnp.abs(points) - half_extents
    positive_offset = jnp.maximum(offset, 0.0)
    outside_norm = jnp.linalg.norm(positive_offset, axis=-1)
    inside = jnp.minimum(jnp.maximum(offset[..., 0], offset[..., 1]), 0.0)
    values = outside_norm + inside

    signs = jnp.sign(points)
    outside_gradients = (
        signs
        * positive_offset
        / jnp.maximum(outside_norm[..., None], 1e-8)
    )
    x_is_nearest = offset[..., 0] >= offset[..., 1]
    inside_gradients = jnp.stack(
        (
            jnp.where(x_is_nearest, signs[..., 0], 0.0),
            jnp.where(x_is_nearest, 0.0, signs[..., 1]),
        ),
        axis=-1,
    )
    gradients = jnp.where(
        (outside_norm > 1e-8)[..., None],
        outside_gradients,
        inside_gradients,
    )
    return values, gradients


def _guard_blend_weight(
    analytic_values: jnp.ndarray,
    *,
    full_distance: float,
    zero_distance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a C1 learned-shell weight and its derivative with respect to h."""
    denominator = jnp.asarray(
        zero_distance - full_distance,
        dtype=NSDF_DTYPE,
    )
    raw = (
        jnp.asarray(zero_distance, dtype=NSDF_DTYPE) - analytic_values
    ) / denominator
    coordinate = jnp.clip(raw, 0.0, 1.0)
    weight = coordinate * coordinate * (3.0 - 2.0 * coordinate)
    interior = (raw > 0.0) & (raw < 1.0)
    derivative = jnp.where(
        interior,
        -6.0 * coordinate * (1.0 - coordinate) / denominator,
        0.0,
    )
    return weight, derivative


def _world_points_to_all_part_frames(
    world_points: jnp.ndarray,
    part_centers: jnp.ndarray,
    part_angles: jnp.ndarray,
) -> jnp.ndarray:
    delta = world_points[:, None, :] - part_centers[None, :, :]
    c = jnp.cos(part_angles)[None, :]
    s = jnp.sin(part_angles)[None, :]
    return jnp.stack(
        (
            delta[..., 0] * c + delta[..., 1] * s,
            -delta[..., 0] * s + delta[..., 1] * c,
        ),
        axis=-1,
    )


def _world_points_to_selected_parts_frames(
    world_points: jnp.ndarray,
    selected_centers: jnp.ndarray,
    selected_angles: jnp.ndarray,
) -> jnp.ndarray:
    delta = (
        world_points[:, None, :, :]
        - selected_centers[:, :, None, :]
    )
    c = jnp.cos(selected_angles)[:, :, None]
    s = jnp.sin(selected_angles)[:, :, None]
    return jnp.stack(
        (
            delta[..., 0] * c + delta[..., 1] * s,
            -delta[..., 0] * s + delta[..., 1] * c,
        ),
        axis=-1,
    )


def _part_poses(state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute rigid-part poses without constructing any Jacobians."""
    state = _validate_state(state)
    base_position = state[:2]
    base_angle = state[2]
    joint_angles = state[3:].reshape((2, 4))
    c = jnp.cos(base_angle)
    s = jnp.sin(base_angle)
    rotation = jnp.stack(
        (
            jnp.stack((c, -s)),
            jnp.stack((s, c)),
        )
    )
    arm_offsets = jnp.asarray(_ARM_OFFSETS, dtype=NSDF_DTYPE)
    shoulders = base_position[None, :] + arm_offsets @ rotation.T
    cumulative_angles = base_angle + jnp.cumsum(joint_angles, axis=1)
    directions = jnp.stack(
        (jnp.cos(cumulative_angles), jnp.sin(cumulative_angles)),
        axis=2,
    )
    lengths = jnp.asarray(_LINK_LENGTHS, dtype=NSDF_DTYPE)
    displacements = directions * lengths[None, :, None]
    preceding = jnp.concatenate(
        (
            jnp.zeros((2, 1, 2), dtype=NSDF_DTYPE),
            jnp.cumsum(displacements[:, :-1, :], axis=1),
        ),
        axis=1,
    )
    link_centers = (
        shoulders[:, None, :]
        + preceding
        + 0.5 * displacements
    )
    centers = jnp.concatenate(
        (base_position[None, :], link_centers.reshape((-1, 2))),
        axis=0,
    )
    angles = jnp.concatenate(
        (base_angle[None], cumulative_angles.reshape(-1)),
        axis=0,
    )
    return centers, angles


def _part_poses_and_jacobians(
    state: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute all rigid-part poses and their exact state Jacobians."""
    state = _validate_state(state)
    base_position = state[:2]
    base_angle = state[2]
    joint_angles = state[3:].reshape((2, 4))

    centers = [base_position]
    angles = [base_angle]

    base_center_jacobian = jnp.zeros(
        (2, MOBILE_ARM_STATE_DIM),
        dtype=NSDF_DTYPE,
    )
    base_center_jacobian = base_center_jacobian.at[0, 0].set(1.0)
    base_center_jacobian = base_center_jacobian.at[1, 1].set(1.0)
    base_angle_jacobian = jnp.zeros(
        (MOBILE_ARM_STATE_DIM,),
        dtype=NSDF_DTYPE,
    ).at[2].set(1.0)
    center_jacobians = [base_center_jacobian]
    angle_jacobians = [base_angle_jacobian]

    c = jnp.cos(base_angle)
    s = jnp.sin(base_angle)
    rotation = jnp.asarray(((c, -s), (s, c)), dtype=NSDF_DTYPE)

    for arm_index, offset_values in enumerate(_ARM_OFFSETS):
        offset = jnp.asarray(offset_values, dtype=NSDF_DTYPE)
        shoulder = base_position + rotation @ offset
        shoulder_jacobian = base_center_jacobian
        rotated_offset_perpendicular = jnp.asarray(
            (
                -s * offset[0] - c * offset[1],
                c * offset[0] - s * offset[1],
            ),
            dtype=NSDF_DTYPE,
        )
        shoulder_jacobian = shoulder_jacobian.at[:, 2].add(
            rotated_offset_perpendicular
        )

        cumulative_angles = base_angle + jnp.cumsum(joint_angles[arm_index])
        directions = jnp.stack(
            (jnp.cos(cumulative_angles), jnp.sin(cumulative_angles)),
            axis=1,
        )
        perpendiculars = jnp.stack(
            (-jnp.sin(cumulative_angles), jnp.cos(cumulative_angles)),
            axis=1,
        )
        lengths = jnp.asarray(_LINK_LENGTHS, dtype=NSDF_DTYPE)
        start = shoulder

        for link_index in range(4):
            link_angle = cumulative_angles[link_index]
            link_center = (
                start + 0.5 * lengths[link_index] * directions[link_index]
            )
            angle_jacobian = jnp.zeros(
                (MOBILE_ARM_STATE_DIM,),
                dtype=NSDF_DTYPE,
            ).at[2].set(1.0)
            joint_start = 3 + 4 * arm_index
            angle_jacobian = angle_jacobian.at[
                joint_start : joint_start + link_index + 1
            ].set(1.0)

            center_jacobian = shoulder_jacobian
            for segment_index in range(link_index):
                derivative = (
                    lengths[segment_index] * perpendiculars[segment_index]
                )
                center_jacobian = center_jacobian.at[:, 2].add(derivative)
                center_jacobian = center_jacobian.at[
                    :,
                    joint_start : joint_start + segment_index + 1,
                ].add(derivative[:, None])

            final_derivative = (
                0.5 * lengths[link_index] * perpendiculars[link_index]
            )
            center_jacobian = center_jacobian.at[:, 2].add(final_derivative)
            center_jacobian = center_jacobian.at[
                :,
                joint_start : joint_start + link_index + 1,
            ].add(final_derivative[:, None])

            centers.append(link_center)
            angles.append(link_angle)
            center_jacobians.append(center_jacobian)
            angle_jacobians.append(angle_jacobian)
            start = start + lengths[link_index] * directions[link_index]

    return (
        jnp.stack(centers),
        jnp.stack(angles),
        jnp.stack(center_jacobians),
        jnp.stack(angle_jacobians),
    )


def _world_to_local_jacobians(
    world_points: jnp.ndarray,
    part_centers: jnp.ndarray,
    part_angles: jnp.ndarray,
    center_jacobians: jnp.ndarray,
    angle_jacobians: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiate selected world-to-part rigid transforms."""
    delta = world_points - part_centers
    c = jnp.cos(part_angles)
    s = jnp.sin(part_angles)
    world_to_local = jnp.stack(
        (
            jnp.stack((c, s), axis=1),
            jnp.stack((-s, c), axis=1),
        ),
        axis=1,
    )
    local_points = jnp.einsum("oij,oj->oi", world_to_local, delta)
    translation_term = -jnp.einsum(
        "oij,ojs->ois",
        world_to_local,
        center_jacobians,
    )
    rotation_derivative = jnp.stack(
        (local_points[:, 1], -local_points[:, 0]),
        axis=1,
    )
    return (
        translation_term
        + rotation_derivative[:, :, None] * angle_jacobians[:, None, :]
    )
