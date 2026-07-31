from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import jax.numpy as jnp
import numpy as np

from sdf.geometry import ObstacleField
from sdf.mobile_arm_sdf import MobileArmSDF
from sdf.nsdf_network import (
    LEGACY_ARCHITECTURE_ORDER,
    NSDF_DTYPE,
    legacy_sdf_values,
    legacy_sdf_values_and_gradients,
    params_to_jax,
    validate_legacy_params,
)


class PretrainedSDFUnavailable(RuntimeError):
    """Raised when no matching pretrained SDF asset is available."""


class PretrainedSDFIntegrityError(PretrainedSDFUnavailable):
    """Raised when a retrained checkpoint or its metadata fails validation."""


@dataclass(frozen=True)
class PretrainedSDFSpec:
    model_file: str
    shape_name: str
    source_shape_index: int | None
    metadata_file: str | None = None
    variant: Literal["legacy", "retrained"] = "legacy"
    expected_width: float | None = None
    expected_height: float | None = None
    geometry_padding: float = 0.0


@dataclass(frozen=True)
class MobileArmPretrainedSDFSpec:
    base_model_file: str
    base_metadata_file: str
    link_model_file: str
    link_metadata_file: str


SUPPORTED_PRETRAINED_SDFS = {
    "unicycle": PretrainedSDFSpec("link1_model_4_16.npy", "rectangle 1.0 x 0.4", 0),
    "dynamic_unicycle": PretrainedSDFSpec("link1_model_4_16.npy", "rectangle 1.0 x 0.4", 0),
    "planar_quadrotor": PretrainedSDFSpec("link7_model_4_16.npy", "quad2 0.56 x 0.28", 6),
}

SUPPORTED_RETRAINED_SDFS = {
    "unicycle": PretrainedSDFSpec(
        "link1_retrained_model_4_16.npy",
        "retrained rectangle 1.0 x 0.4",
        None,
        metadata_file="link1_retrained_model_4_16.json",
        variant="retrained",
        expected_width=1.0,
        expected_height=0.4,
        geometry_padding=0.08,
    ),
    "dynamic_unicycle": PretrainedSDFSpec(
        "link1_retrained_model_4_16.npy",
        "retrained rectangle 1.0 x 0.4",
        None,
        metadata_file="link1_retrained_model_4_16.json",
        variant="retrained",
        expected_width=1.0,
        expected_height=0.4,
        geometry_padding=0.08,
    ),
    "planar_quadrotor": PretrainedSDFSpec(
        "link7_retrained_model_4_16.npy",
        "retrained rectangle 0.56 x 0.28",
        None,
        metadata_file="link7_retrained_model_4_16.json",
        variant="retrained",
        expected_width=0.56,
        expected_height=0.28,
        geometry_padding=0.06,
    ),
}

MOBILE_ARM_RETRAINED_SDF = MobileArmPretrainedSDFSpec(
    base_model_file="mobile_arm_base_model_4_16.npy",
    base_metadata_file="mobile_arm_base_model_4_16.json",
    link_model_file="mobile_arm_link_model_4_16.npy",
    link_metadata_file="mobile_arm_link_model_4_16.json",
)

PINNED_RETRAINED_CHECKPOINT_SHA256 = {
    "link1_retrained_model_4_16.npy": (
        "0fbc36d37475ab8ff9c205ff2c200b3d078587b6aa7ff299a1b10f55acbcc6e4"
    ),
    "link7_retrained_model_4_16.npy": (
        "da1acd5be2501f55d5d9f8784825220990f6b82d70977dd696e98953242f5bec"
    ),
    "mobile_arm_base_model_4_16.npy": (
        "e8f6dd65ebe4acf0fe29dca076fa8aeee1bae3dd50b30d0ce07557df28b5fc30"
    ),
    "mobile_arm_link_model_4_16.npy": (
        "ea1bddcd141a1ffe1c9a8ea61fb7254514baee6f1787d4690e62c1b2c845a095"
    ),
}

UNSUPPORTED_PRETRAINED_SDFS = {
    "single_integrator": "no pretrained disk/point SDF was found in sdf/trained_models",
    "mobile_arm": (
        "the available pretrained files are canonical single-shape SDFs, not a model for "
        "the original mobile-arm base plus two fixed 4-link arms"
    ),
}


@dataclass
class PretrainedShapeSDF:
    """JAX evaluator for the vendored Flax SDFNet checkpoints."""

    robot_name: str
    spec: PretrainedSDFSpec
    params: dict
    points_per_obstacle: int = 16
    metadata: Mapping[str, Any] | None = None
    geometry_padding: float = 0.0
    model_margin: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.geometry_padding) or self.geometry_padding < 0.0:
            raise ValueError("geometry_padding must be finite and nonnegative")
        if not math.isfinite(self.model_margin) or self.model_margin < 0.0:
            raise ValueError("model_margin must be finite and nonnegative")
        self.params = _params_to_jax(self.params)
        self._field_cache_key: tuple[int, tuple[tuple[float, float, float], ...]] | None = None
        self._field_cache_points: np.ndarray | None = None

    @property
    def description(self) -> str:
        source = (
            f", source shape index {self.spec.source_shape_index}"
            if self.spec.source_shape_index is not None
            else ""
        )
        margins = (
            f", geometry_padding={self.geometry_padding:g}, "
            f"model_margin={self.model_margin:g}"
            if self.spec.variant == "retrained"
            else ""
        )
        return (
            f"{self.spec.model_file} ({self.spec.shape_name}{source}, "
            f"variant={self.spec.variant}{margins})"
        )

    @property
    def total_margin(self) -> float:
        return self.geometry_padding + self.model_margin

    @property
    def provenance(self) -> dict[str, Any]:
        metadata_provenance = (
            dict(self.metadata.get("provenance", {}))
            if isinstance(self.metadata, Mapping)
            and isinstance(self.metadata.get("provenance"), Mapping)
            else {}
        )
        training = self.metadata.get("training") if isinstance(self.metadata, Mapping) else None
        training_config = (
            training.get("config")
            if isinstance(training, Mapping) and isinstance(training.get("config"), Mapping)
            else {}
        )
        checkpoint = (
            self.metadata.get("checkpoint")
            if isinstance(self.metadata, Mapping)
            and isinstance(self.metadata.get("checkpoint"), Mapping)
            else {}
        )
        return {
            "variant": self.spec.variant,
            "checkpoint": self.spec.model_file,
            "checkpoint_sha256": checkpoint.get("sha256"),
            "metadata_file": self.spec.metadata_file,
            "training_seed": training_config.get("seed"),
            "training_provenance": metadata_provenance,
            "geometry_padding": self.geometry_padding,
            "model_margin": self.model_margin,
            "total_margin": self.total_margin,
        }

    def signed_distance(self, points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the checkpoint in its local shape frame."""
        local_points = jnp.atleast_2d(jnp.asarray(points, dtype=NSDF_DTYPE))
        local_points = _pad_points_to_3d(local_points)
        return self._values(local_points)

    def obstacle_barriers(self, state: jnp.ndarray, field: ObstacleField) -> jnp.ndarray:
        """Return one robot-shape SDF barrier value per obstacle."""
        obstacle_points = self._obstacle_points(field)
        state = jnp.asarray(state, dtype=NSDF_DTYPE)
        x, y = state[:2]
        theta = state[2] if state.shape[0] >= 3 else jnp.array(0.0, dtype=NSDF_DTYPE)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=NSDF_DTYPE)
        local_xy = (obstacle_points - jnp.array([x, y], dtype=NSDF_DTYPE)) @ rot
        local_points = jnp.column_stack((local_xy, jnp.zeros(local_xy.shape[0], dtype=NSDF_DTYPE)))
        values = self._values(local_points)
        values = values.reshape((len(field.obstacles), self.points_per_obstacle))
        return jnp.min(values, axis=1) - jnp.asarray(self.total_margin, dtype=NSDF_DTYPE)

    def obstacle_barriers_and_jacobian(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return per-obstacle barriers and their exact robot-state Jacobian.

        The network and its local gradients are evaluated once. The selected
        surface point for each obstacle is then differentiated through the
        world-to-robot rigid transform. State components after ``(x, y, yaw)``
        do not directly affect the shape SDF and therefore have zero columns.
        """
        obstacle_points = self._obstacle_points(field)
        state = jnp.asarray(state, dtype=NSDF_DTYPE)
        x, y = state[:2]
        theta = state[2] if state.shape[0] >= 3 else jnp.array(0.0, dtype=NSDF_DTYPE)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=NSDF_DTYPE)
        local_xy = (obstacle_points - jnp.array([x, y], dtype=NSDF_DTYPE)) @ rot
        local_points = jnp.column_stack((local_xy, jnp.zeros(local_xy.shape[0], dtype=NSDF_DTYPE)))
        point_values, point_local_gradients = self._value_and_local_gradient(local_points)

        obstacle_count = len(field.obstacles)
        values = point_values.reshape((obstacle_count, self.points_per_obstacle))
        local_gradients = point_local_gradients[:, :2].reshape(
            (obstacle_count, self.points_per_obstacle, 2)
        )
        local_coordinates = local_xy.reshape((obstacle_count, self.points_per_obstacle, 2))
        nearest = jnp.argmin(values, axis=1)
        row = jnp.arange(obstacle_count)
        chosen_values = values[row, nearest] - jnp.asarray(
            self.total_margin,
            dtype=NSDF_DTYPE,
        )
        chosen_gradients = local_gradients[row, nearest]
        chosen_coordinates = local_coordinates[row, nearest]

        gx, gy = chosen_gradients[:, 0], chosen_gradients[:, 1]
        local_x, local_y = chosen_coordinates[:, 0], chosen_coordinates[:, 1]
        jacobian = jnp.zeros((obstacle_count, state.shape[0]), dtype=NSDF_DTYPE)
        jacobian = jacobian.at[:, 0].set(-gx * c + gy * s)
        jacobian = jacobian.at[:, 1].set(-gx * s - gy * c)
        if state.shape[0] >= 3:
            jacobian = jacobian.at[:, 2].set(gx * local_y - gy * local_x)
        return chosen_values, jacobian

    def _obstacle_points(self, field: ObstacleField) -> jnp.ndarray:
        obstacle_key = tuple(
            (float(center[0]), float(center[1]), obstacle.radius)
            for obstacle in field.obstacles
            for center in (np.asarray(obstacle.center, dtype=np.float64),)
        )
        cache_key = (self.points_per_obstacle, obstacle_key)
        if cache_key != self._field_cache_key:
            theta = np.linspace(
                0.0,
                2.0 * np.pi,
                self.points_per_obstacle,
                endpoint=False,
            )
            unit_circle = np.stack((np.cos(theta), np.sin(theta)), axis=1)
            centers = np.asarray([[x, y] for x, y, _radius in obstacle_key], dtype=np.float64)
            radii = np.asarray([radius for _x, _y, radius in obstacle_key], dtype=np.float64)
            self._field_cache_key = cache_key
            self._field_cache_points = (
                centers[:, None, :] + radii[:, None, None] * unit_circle[None, :, :]
            ).reshape((-1, 2)).astype(np.float32)
        assert self._field_cache_points is not None
        return jnp.asarray(self._field_cache_points, dtype=NSDF_DTYPE)

    def _values(self, points: jnp.ndarray) -> jnp.ndarray:
        return legacy_sdf_values(self.params, points)

    def _value_and_local_gradient(self, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return legacy_sdf_values_and_gradients(self.params, points)


class PretrainedMobileArmSDF(MobileArmSDF):
    """Mobile-arm NSDF with structured production-checkpoint provenance."""

    @property
    def description(self) -> str:
        return (
            f"compositional mobile-arm NSDF "
            f"(base={self.base_description}; link={self.link_description}; "
            f"mode={self.obstacle_mode}; geometry_padding={self.geometry_padding:g}; "
            f"base_model_margin={self.base_model_margin:g}; "
            f"link_model_margin={self.link_model_margin:g} canonical/scale-adjusted; "
            f"broad_phase_parts={self.broad_phase_parts}; "
            f"narrow_phase={self.narrow_phase_points}/{self.points_per_obstacle}; "
            f"analytic_guard={self.analytic_guard}; "
            f"learned_blend={self.learned_blend_full_distance:g}"
            f"..{self.learned_blend_zero_distance:g})"
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "variant": "retrained",
            "geometry_padding": self.geometry_padding,
            "global_model_margin": self.model_margin,
            "base": _primitive_provenance(
                self.base_metadata,
                metadata_file=MOBILE_ARM_RETRAINED_SDF.base_metadata_file,
                model_margin=self.base_model_margin,
                scale_rule="unscaled",
            ),
            "link": _primitive_provenance(
                self.link_metadata,
                metadata_file=MOBILE_ARM_RETRAINED_SDF.link_metadata_file,
                model_margin=self.link_model_margin,
                scale_rule="physical margin = link scale * canonical model margin",
            ),
            "runtime": {
                "obstacle_mode": self.obstacle_mode,
                "points_per_obstacle": self.points_per_obstacle,
                "narrow_phase_points": self.narrow_phase_points,
                "broad_phase_parts": self.broad_phase_parts,
                "analytic_guard": self.analytic_guard,
                "learned_blend_full_distance": (
                    self.learned_blend_full_distance
                ),
                "learned_blend_zero_distance": (
                    self.learned_blend_zero_distance
                ),
            },
            "description": self.description,
        }


def load_pretrained_sdf_for_robot(
    robot_name: str,
    *,
    repo_root: Path | None = None,
    variant: Literal["legacy", "retrained"] = "legacy",
    mobile_arm_obstacle_mode: Literal[
        "narrow_surface",
        "circle_center",
        "surface_points",
    ] = "narrow_surface",
    mobile_arm_points_per_obstacle: int = 64,
    mobile_arm_narrow_phase_points: int = 4,
    mobile_arm_broad_phase_parts: int = 2,
    mobile_arm_learned_blend_full_distance: float = 0.05,
    mobile_arm_learned_blend_zero_distance: float = 0.20,
) -> PretrainedShapeSDF | PretrainedMobileArmSDF:
    """Load a legacy or checksum-verified retrained robot NSDF.

    ``legacy`` remains the default and preserves the previous numerical
    behavior. Retrained rigid-shape barriers subtract the robot's analytic
    body-point padding and the validation-calibrated model margin. Raw
    ``signed_distance`` queries remain unshifted.
    """
    if variant not in ("legacy", "retrained"):
        raise ValueError("variant must be either 'legacy' or 'retrained'")
    root = repo_root or Path(__file__).resolve().parents[1]
    if robot_name == "mobile_arm" and variant == "retrained":
        return _load_retrained_mobile_arm(
            root,
            obstacle_mode=mobile_arm_obstacle_mode,
            points_per_obstacle=mobile_arm_points_per_obstacle,
            narrow_phase_points=mobile_arm_narrow_phase_points,
            broad_phase_parts=mobile_arm_broad_phase_parts,
            learned_blend_full_distance=(
                mobile_arm_learned_blend_full_distance
            ),
            learned_blend_zero_distance=(
                mobile_arm_learned_blend_zero_distance
            ),
        )
    if variant == "legacy":
        if robot_name in UNSUPPORTED_PRETRAINED_SDFS:
            raise PretrainedSDFUnavailable(UNSUPPORTED_PRETRAINED_SDFS[robot_name])
        specs = SUPPORTED_PRETRAINED_SDFS
    else:
        if robot_name == "single_integrator":
            raise PretrainedSDFUnavailable(
                "no retrained disk/point SDF is available for single_integrator"
            )
        specs = SUPPORTED_RETRAINED_SDFS
    if robot_name not in specs:
        raise PretrainedSDFUnavailable(
            f"no {variant} pretrained SDF mapping is configured for robot '{robot_name}'"
        )

    spec = specs[robot_name]
    if variant == "retrained":
        params, metadata = _load_verified_retrained_checkpoint(root, spec)
        model_margin = _recommended_model_margin(metadata)
    else:
        params = _load_legacy_checkpoint(root, spec)
        metadata = None
        model_margin = 0.0
    return PretrainedShapeSDF(
        robot_name=robot_name,
        spec=spec,
        params=params,
        metadata=metadata,
        geometry_padding=spec.geometry_padding,
        model_margin=model_margin,
    )


def _load_legacy_checkpoint(root: Path, spec: PretrainedSDFSpec) -> dict:
    model_path = root / "sdf" / "trained_models" / spec.model_file
    if not model_path.exists():
        raise PretrainedSDFUnavailable(f"missing pretrained model file: {model_path}")
    return np.load(model_path, allow_pickle=True).item()


def _load_verified_retrained_checkpoint(
    root: Path,
    spec: PretrainedSDFSpec,
) -> tuple[dict, dict[str, Any]]:
    if spec.metadata_file is None:
        raise PretrainedSDFIntegrityError(
            f"retrained checkpoint {spec.model_file} has no metadata mapping"
        )
    models_dir = root / "sdf" / "trained_models"
    model_path = models_dir / spec.model_file
    metadata_path = models_dir / spec.metadata_file
    if not model_path.exists():
        raise PretrainedSDFUnavailable(f"missing retrained model file: {model_path}")
    if not metadata_path.exists():
        raise PretrainedSDFIntegrityError(
            f"missing retrained metadata file: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PretrainedSDFIntegrityError(
            f"invalid retrained metadata file: {metadata_path}"
        ) from exc
    if not isinstance(metadata, dict):
        raise PretrainedSDFIntegrityError(
            f"retrained metadata must be a JSON object: {metadata_path}"
        )
    _validate_retrained_metadata(metadata, spec, model_path)
    try:
        params = np.load(model_path, allow_pickle=True).item()
        validate_legacy_params(params, expected_hidden_dim=16)
    except Exception as exc:
        raise PretrainedSDFIntegrityError(
            f"invalid retrained parameter checkpoint: {model_path}"
        ) from exc
    return params, metadata


def _validate_retrained_metadata(
    metadata: Mapping[str, Any],
    spec: PretrainedSDFSpec,
    model_path: Path,
) -> None:
    if metadata.get("format_version") != 1:
        raise PretrainedSDFIntegrityError(
            f"unsupported metadata format for {model_path.name}"
        )
    checkpoint = metadata.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise PretrainedSDFIntegrityError(
            f"metadata for {model_path.name} is missing checkpoint information"
        )
    if checkpoint.get("file") != model_path.name:
        raise PretrainedSDFIntegrityError(
            f"metadata checkpoint filename does not match {model_path.name}"
        )
    expected_checksum = checkpoint.get("sha256")
    if not isinstance(expected_checksum, str) or len(expected_checksum) != 64:
        raise PretrainedSDFIntegrityError(
            f"metadata for {model_path.name} has an invalid SHA-256"
        )
    pinned_checksum = PINNED_RETRAINED_CHECKPOINT_SHA256.get(model_path.name)
    if pinned_checksum is None:
        raise PretrainedSDFIntegrityError(
            f"no trusted SHA-256 is pinned for retrained checkpoint {model_path.name}"
        )
    if expected_checksum != pinned_checksum:
        raise PretrainedSDFIntegrityError(
            f"metadata SHA-256 does not match the trusted pin for {model_path.name}"
        )
    actual_checksum = _sha256(model_path)
    if actual_checksum != pinned_checksum:
        raise PretrainedSDFIntegrityError(
            f"SHA-256 mismatch for retrained checkpoint {model_path.name}"
        )

    architecture = metadata.get("architecture")
    if (
        not isinstance(architecture, Mapping)
        or architecture.get("input_dim") != 3
        or architecture.get("hidden_dim") != 16
        or architecture.get("output_dim") != 1
        or architecture.get("dtype") != "float32"
        or architecture.get("name") != "legacy_sdfnet_4x16"
        or architecture.get("layer_order") != list(LEGACY_ARCHITECTURE_ORDER)
    ):
        raise PretrainedSDFIntegrityError(
            f"metadata architecture does not match legacy 4x16 float32 for {model_path.name}"
        )
    geometry = metadata.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("type") != "axis_aligned_rectangle":
        raise PretrainedSDFIntegrityError(
            f"metadata geometry is not an axis-aligned rectangle for {model_path.name}"
        )
    try:
        width = float(geometry["width"])
        height = float(geometry["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PretrainedSDFIntegrityError(
            f"metadata geometry dimensions are invalid for {model_path.name}"
        ) from exc
    if (
        spec.expected_width is not None
        and not math.isclose(width, spec.expected_width, abs_tol=1e-6)
    ) or (
        spec.expected_height is not None
        and not math.isclose(height, spec.expected_height, abs_tol=1e-6)
    ):
        raise PretrainedSDFIntegrityError(
            f"metadata geometry dimensions do not match {spec.shape_name}"
        )
    _recommended_model_margin(metadata)


def _recommended_model_margin(metadata: Mapping[str, Any]) -> float:
    validation = metadata.get("validation")
    if not isinstance(validation, Mapping):
        raise PretrainedSDFIntegrityError(
            "retrained metadata is missing validation metrics"
        )
    try:
        margin = float(validation["recommended_conservative_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PretrainedSDFIntegrityError(
            "retrained metadata has no numeric recommended conservative margin"
        ) from exc
    if not math.isfinite(margin) or margin < 0.0:
        raise PretrainedSDFIntegrityError(
            "retrained metadata conservative margin must be finite and nonnegative"
        )
    return margin


def _load_retrained_mobile_arm(
    root: Path,
    *,
    obstacle_mode: Literal[
        "narrow_surface",
        "circle_center",
        "surface_points",
    ],
    points_per_obstacle: int,
    narrow_phase_points: int,
    broad_phase_parts: int,
    learned_blend_full_distance: float,
    learned_blend_zero_distance: float,
) -> PretrainedMobileArmSDF:
    mobile_spec = MOBILE_ARM_RETRAINED_SDF
    base_spec = PretrainedSDFSpec(
        mobile_spec.base_model_file,
        "mobile-arm base 1.5 x 1.0",
        None,
        metadata_file=mobile_spec.base_metadata_file,
        variant="retrained",
        expected_width=1.5,
        expected_height=1.0,
        geometry_padding=0.02,
    )
    link_spec = PretrainedSDFSpec(
        mobile_spec.link_model_file,
        "mobile-arm canonical link 0.8 x 0.2",
        None,
        metadata_file=mobile_spec.link_metadata_file,
        variant="retrained",
        expected_width=0.8,
        expected_height=0.2,
        geometry_padding=0.02,
    )
    base_params, base_metadata = _load_verified_retrained_checkpoint(root, base_spec)
    link_params, link_metadata = _load_verified_retrained_checkpoint(root, link_spec)
    base_model_margin = _recommended_model_margin(base_metadata)
    link_model_margin = _recommended_model_margin(link_metadata)
    return PretrainedMobileArmSDF(
        base_params=base_params,
        link_params=link_params,
        base_metadata=base_metadata,
        link_metadata=link_metadata,
        base_description=base_spec.model_file,
        link_description=link_spec.model_file,
        geometry_padding=0.02,
        model_margin=0.0,
        base_model_margin=base_model_margin,
        link_model_margin=link_model_margin,
        obstacle_mode=obstacle_mode,
        points_per_obstacle=points_per_obstacle,
        narrow_phase_points=narrow_phase_points,
        broad_phase_parts=broad_phase_parts,
        learned_blend_full_distance=learned_blend_full_distance,
        learned_blend_zero_distance=learned_blend_zero_distance,
    )


def _primitive_provenance(
    metadata: Mapping[str, Any] | None,
    *,
    metadata_file: str,
    model_margin: float,
    scale_rule: str,
) -> dict[str, Any]:
    checkpoint = (
        metadata.get("checkpoint")
        if isinstance(metadata, Mapping)
        and isinstance(metadata.get("checkpoint"), Mapping)
        else {}
    )
    training = metadata.get("training") if isinstance(metadata, Mapping) else None
    training_config = (
        training.get("config")
        if isinstance(training, Mapping) and isinstance(training.get("config"), Mapping)
        else {}
    )
    metadata_provenance = (
        metadata.get("provenance")
        if isinstance(metadata, Mapping)
        and isinstance(metadata.get("provenance"), Mapping)
        else {}
    )
    return {
        "checkpoint": checkpoint.get("file"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "metadata_file": metadata_file,
        "training_seed": training_config.get("seed"),
        "training_provenance": dict(metadata_provenance),
        "model_margin": model_margin,
        "margin_scale_rule": scale_rule,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _params_to_jax(params: dict) -> dict:
    return params_to_jax(params)


def _pad_points_to_3d(points: jnp.ndarray) -> jnp.ndarray:
    if points.shape[1] == 2:
        return jnp.column_stack((points, jnp.zeros(points.shape[0], dtype=NSDF_DTYPE)))
    return points
