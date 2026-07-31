from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sdf.mobile_arm_sdf as mobile_arm_sdf
from controller import MPPIConfig, MPPIController
from robots import create_robot
from robots.mobile_arm import MobileArmRobot
from sdf.geometry import CircleObstacle, ObstacleField, default_obstacle_field
from sdf.mobile_arm_sdf import MobileArmSDF
from sdf.pretrained_sdf import load_pretrained_sdf_for_robot


BASE_PARAMS = {"half_extents": (0.75, 0.5)}
LINK_PARAMS = {"half_extents": (0.4, 0.1)}


@pytest.fixture
def exact_rectangle_primitives(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only the primitive networks; retain the composition under test."""

    def rectangle_values(params, points):
        points = jnp.atleast_2d(jnp.asarray(points, dtype=jnp.float32))
        half_extents = jnp.asarray(params["half_extents"], dtype=jnp.float32)
        q = jnp.abs(points[:, :2]) - half_extents
        outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=1)
        inside = jnp.minimum(jnp.maximum(q[:, 0], q[:, 1]), 0.0)
        return outside + inside

    def rectangle_values_and_gradients(params, points):
        points = jnp.atleast_2d(jnp.asarray(points, dtype=jnp.float32))
        half_extents = jnp.asarray(params["half_extents"], dtype=jnp.float32)

        def one_value(point):
            q = jnp.abs(point[:2]) - half_extents
            outside = jnp.linalg.norm(jnp.maximum(q, 0.0))
            inside = jnp.minimum(jnp.maximum(q[0], q[1]), 0.0)
            return outside + inside

        return jax.vmap(jax.value_and_grad(one_value))(points)

    monkeypatch.setattr(mobile_arm_sdf, "params_to_jax", lambda params: params)
    monkeypatch.setattr(
        mobile_arm_sdf,
        "validate_legacy_params",
        lambda _params: None,
    )
    monkeypatch.setattr(mobile_arm_sdf, "legacy_sdf_values", rectangle_values)
    monkeypatch.setattr(
        mobile_arm_sdf,
        "legacy_sdf_values_and_gradients",
        rectangle_values_and_gradients,
    )


def test_part_poses_and_jacobians_match_robot_forward_kinematics() -> None:
    robot = MobileArmRobot()
    state = _test_state()

    centers, angles, center_jacobians, angle_jacobians = (
        mobile_arm_sdf._part_poses_and_jacobians(state)
    )
    base_polygon = robot.base_polygon(state)
    link_polygons = robot.link_polygons(state)
    expected_centers = jnp.concatenate(
        (
            jnp.mean(base_polygon, axis=0, keepdims=True),
            jnp.mean(link_polygons, axis=1),
        ),
        axis=0,
    )
    arm_segments = robot.arm_polylines(state)
    segment_vectors = arm_segments[:, 1:, :] - arm_segments[:, :-1, :]
    expected_link_angles = jnp.arctan2(
        segment_vectors[:, :, 1],
        segment_vectors[:, :, 0],
    ).reshape(-1)

    autodiff_center_jacobians = jax.jacfwd(
        lambda candidate: mobile_arm_sdf._part_poses(candidate)[0]
    )(state)
    autodiff_angle_jacobians = jax.jacfwd(
        lambda candidate: mobile_arm_sdf._part_poses(candidate)[1]
    )(state)

    assert centers.shape == (9, 2)
    assert angles.shape == (9,)
    assert center_jacobians.shape == (9, 2, 11)
    assert angle_jacobians.shape == (9, 11)
    assert jnp.allclose(centers, expected_centers, atol=2e-6)
    assert jnp.allclose(
        jnp.exp(1j * angles[1:]),
        jnp.exp(1j * expected_link_angles),
        atol=2e-6,
    )
    assert jnp.allclose(
        center_jacobians,
        autodiff_center_jacobians,
        rtol=2e-5,
        atol=2e-6,
    )
    assert jnp.allclose(
        angle_jacobians,
        autodiff_angle_jacobians,
        rtol=2e-5,
        atol=2e-6,
    )


def test_composite_barriers_match_exact_rectangle_union_distances(
    exact_rectangle_primitives,
) -> None:
    robot = MobileArmRobot()
    state = _test_state()
    field = ObstacleField(
        obstacles=(
            CircleObstacle(center=(-0.9, 1.4), radius=0.13),
            CircleObstacle(center=(1.7, 0.8), radius=0.21),
            CircleObstacle(center=(0.3, -1.6), radius=0.08),
            CircleObstacle(center=(2.1, -0.9), radius=0.17),
        )
    )
    model = MobileArmSDF(
        BASE_PARAMS,
        LINK_PARAMS,
        geometry_padding=0.02,
        model_margin=0.015,
        base_domain_padding=1.0,
        link_domain_padding=1.0,
    )

    actual = model.obstacle_barriers(state, field)
    polygons = jnp.concatenate(
        (
            robot.base_polygon(state)[None, :, :],
            robot.link_polygons(state),
        ),
        axis=0,
    )
    centers = field.centers
    per_part = jax.vmap(
        lambda point: jax.vmap(
            lambda polygon: _oriented_rectangle_sdf(point, polygon)
        )(polygons)
    )(centers)
    exact_clearance = (
        jnp.min(per_part, axis=1)
        - field.radii
        - model.geometry_padding
    )

    assert actual.dtype == jnp.float32
    # The analytic guard prevents learned overestimation. With an exact
    # primitive, the learned branch can only lower clearance by its explicit
    # model margin.
    assert jnp.all(actual <= exact_clearance + 2e-6)
    assert jnp.all(
        actual >= exact_clearance - model.model_margin - 2e-6
    )


def test_direct_full_state_jacobian_matches_autodiff_away_from_ties(
    exact_rectangle_primitives,
) -> None:
    robot = MobileArmRobot()
    state = _test_state()
    polygons = jnp.concatenate(
        (
            robot.base_polygon(state)[None, :, :],
            robot.link_polygons(state),
        ),
        axis=0,
    )
    # Use the base and two separated distal links. Offsetting from the middle
    # of an edge avoids both box corners and nearest-part ties.
    selected_parts = (0, 4, 8)
    obstacle_centers = []
    for part_index in selected_parts:
        polygon = polygons[part_index]
        center = jnp.mean(polygon, axis=0)
        longitudinal = polygon[1] - polygon[0]
        longitudinal = longitudinal / jnp.linalg.norm(longitudinal)
        normal = jnp.asarray(
            (-longitudinal[1], longitudinal[0]),
            dtype=jnp.float32,
        )
        half_width = 0.5 * jnp.linalg.norm(polygon[3] - polygon[0])
        obstacle_centers.append(
            center + 0.07 * longitudinal + (half_width + 0.24) * normal
        )
    field = ObstacleField(
        obstacles=tuple(
            CircleObstacle(
                center=(float(center[0]), float(center[1])),
                radius=0.06,
            )
            for center in obstacle_centers
        )
    )
    model = MobileArmSDF(
        BASE_PARAMS,
        LINK_PARAMS,
        geometry_padding=0.02,
        model_margin=0.01,
        base_domain_padding=1.0,
        link_domain_padding=1.0,
    )

    values, direct_jacobian = model.obstacle_barriers_and_jacobian(state, field)
    autodiff_jacobian = jax.jacfwd(
        lambda candidate: model.obstacle_barriers(candidate, field)
    )(state)

    assert values.shape == (3,)
    assert direct_jacobian.shape == (3, 11)
    assert jnp.all(jnp.isfinite(direct_jacobian))
    assert jnp.allclose(
        direct_jacobian,
        autodiff_jacobian,
        rtol=3e-5,
        atol=4e-6,
    )
    # At least one arm obstacle must exercise joint-state derivatives.
    assert float(jnp.max(jnp.abs(direct_jacobian[:, 3:]))) > 0.05


def test_circle_center_fast_path_matches_dense_surface_reference(
    exact_rectangle_primitives,
) -> None:
    state = _test_state()
    field = ObstacleField(
        obstacles=(CircleObstacle(center=(0.1, 3.8), radius=0.25),)
    )
    fast_model = MobileArmSDF(
        BASE_PARAMS,
        LINK_PARAMS,
        obstacle_mode="circle_center",
    )
    reference_model = MobileArmSDF(
        BASE_PARAMS,
        LINK_PARAMS,
        obstacle_mode="surface_points",
        points_per_obstacle=512,
    )

    fast_value = fast_model.obstacle_barriers(state, field)
    reference_value = reference_model.obstacle_barriers(state, field)

    assert jnp.allclose(fast_value, reference_value, atol=3e-4)


def test_empty_field_and_metadata_validation(
    exact_rectangle_primitives,
) -> None:
    valid_base_metadata = {
        "geometry": {
            "type": "axis_aligned_rectangle",
            "width": 1.5,
            "height": 1.0,
        }
    }
    valid_link_metadata = {
        "geometry": {
            "type": "axis_aligned_rectangle",
            "width": 0.8,
            "height": 0.2,
        }
    }
    model = MobileArmSDF(
        BASE_PARAMS,
        LINK_PARAMS,
        base_metadata=valid_base_metadata,
        link_metadata=valid_link_metadata,
        base_domain_padding=1.0,
        link_domain_padding=1.0,
    )

    values, jacobian = model.obstacle_barriers_and_jacobian(
        _test_state(),
        ObstacleField(obstacles=()),
    )

    assert values.shape == (0,)
    assert jacobian.shape == (0, 11)
    with pytest.raises(ValueError, match="requires either explicit"):
        MobileArmSDF(BASE_PARAMS, LINK_PARAMS)
    with pytest.raises(ValueError, match="base checkpoint geometry"):
        MobileArmSDF(
            BASE_PARAMS,
            LINK_PARAMS,
            base_metadata={
                "geometry": {
                    "type": "axis_aligned_rectangle",
                    "width": 1.0,
                    "height": 1.5,
                }
            },
        )


def test_production_narrow_phase_is_conservative_against_continuous_boxes(
    exact_rectangle_primitives,
) -> None:
    robot = MobileArmRobot()
    rng = np.random.default_rng(23)
    maximum_overestimation = -np.inf

    for _case in range(8):
        state = jnp.asarray(
            (
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-jnp.pi, jnp.pi),
                *rng.uniform(-1.2, 1.2, size=8),
            ),
            dtype=jnp.float32,
        )
        centers = rng.uniform(-4.0, 4.0, size=(12, 2))
        radii = rng.uniform(0.28, 0.62, size=12)
        field = ObstacleField(
            obstacles=tuple(
                CircleObstacle(
                    center=(float(center[0]), float(center[1])),
                    radius=float(radius),
                )
                for center, radius in zip(centers, radii, strict=True)
            )
        )
        model = MobileArmSDF(
            BASE_PARAMS,
            LINK_PARAMS,
            geometry_padding=0.02,
            model_margin=0.01,
            base_domain_padding=1.0,
            link_domain_padding=1.0,
            points_per_obstacle=64,
            narrow_phase_points=4,
        )

        actual = model.obstacle_barriers(state, field)
        exact = _continuous_rectangle_clearances(robot, state, field)
        maximum_overestimation = max(
            maximum_overestimation,
            float(jnp.max(actual - exact)),
        )

        assert jnp.all(jnp.isfinite(actual))
        assert jnp.all(actual <= exact + 3e-6)
        assert not bool(jnp.any((actual > 0.0) & (exact <= 0.0)))

    assert maximum_overestimation <= 3e-6


def test_production_masks_out_of_domain_queries_and_falls_back_to_exact(
    exact_rectangle_primitives,
) -> None:
    state = _test_state()
    field = ObstacleField(
        obstacles=(CircleObstacle(center=(30.0, -25.0), radius=0.62),)
    )
    model = MobileArmSDF(
        BASE_PARAMS,
        LINK_PARAMS,
        base_domain_padding=0.5,
        link_domain_padding=0.5,
    )

    context = model._narrow_phase_context(state, field)
    exact = _continuous_rectangle_clearances(MobileArmRobot(), state, field)
    values, direct_jacobian = model.obstacle_barriers_and_jacobian(
        state,
        field,
    )
    autodiff_jacobian = jax.jacfwd(
        lambda candidate: model.obstacle_barriers(candidate, field)
    )(state)

    assert not bool(context["use_learned"][0])
    assert jnp.allclose(values, exact, atol=3e-6)
    assert jnp.all(jnp.isfinite(direct_jacobian))
    assert jnp.allclose(
        direct_jacobian,
        autodiff_jacobian,
        rtol=3e-5,
        atol=4e-6,
    )
    selected_domains = model._selected_domain_half_extents(
        context["broad_part_indices"]
    )
    assert jnp.all(
        jnp.abs(context["guarded_points"])
        <= selected_domains[:, :, None, :]
    )


def test_promoted_mobile_checkpoints_are_jittable_and_match_autodiff() -> None:
    state = _test_state()
    field = ObstacleField(
        obstacles=(
            CircleObstacle(center=(1.7, 0.8), radius=0.21),
            CircleObstacle(center=(0.3, -1.6), radius=0.08),
            CircleObstacle(center=(2.1, -0.9), radius=0.17),
        )
    )
    model = load_pretrained_sdf_for_robot(
        "mobile_arm",
        variant="retrained",
    )

    evaluate = jax.jit(
        lambda candidate: model.obstacle_barriers_and_jacobian(
            candidate,
            field,
        )
    )
    values, direct_jacobian = evaluate(state)
    autodiff_jacobian = jax.jacfwd(
        lambda candidate: model.obstacle_barriers(candidate, field)
    )(state)
    context = model._narrow_phase_context(state, field)

    assert isinstance(model, MobileArmSDF)
    assert bool(
        jnp.any(
            context["use_learned"]
            & (context["blend_weight"] > 0.0)
        )
    )
    assert values.shape == (3,)
    assert direct_jacobian.shape == (3, 11)
    assert jnp.all(jnp.isfinite(direct_jacobian))
    assert jnp.allclose(
        direct_jacobian,
        autodiff_jacobian,
        rtol=2e-2,
        atol=5e-3,
    )


def test_promoted_mobile_nsdf_runs_tiny_brmppi_command() -> None:
    robot = create_robot("mobile_arm")
    field = default_obstacle_field("mobile_arm")
    model = load_pretrained_sdf_for_robot(
        "mobile_arm",
        variant="retrained",
    )
    controller = MPPIController(
        robot,
        field,
        algo="brmppi",
        sdf_model=model,
        config=MPPIConfig(
            horizon=2,
            samples=3,
            dt=0.1,
            plot_samples=0,
        ),
        seed=7,
    )

    action, diagnostics = controller.command(
        robot.default_state,
        robot.default_goal,
    )

    assert action.shape == (robot.control_dim,)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert diagnostics["barrier_source"] == "neural_sdf"


def test_promoted_guard_has_no_large_fallback_jump_on_fine_sweep() -> None:
    robot = create_robot("mobile_arm")
    model = load_pretrained_sdf_for_robot(
        "mobile_arm",
        variant="retrained",
    )
    step = 0.005
    lines = []
    obstacles = []
    for y in (-1.5, 0.0, 1.5):
        x_coordinates = np.arange(-2.5, 3.5 + 0.5 * step, step)
        lines.append(x_coordinates.shape[0])
        obstacles.extend(
            CircleObstacle(center=(float(x), y), radius=0.45)
            for x in x_coordinates
        )
    field = ObstacleField(obstacles=tuple(obstacles))

    values = np.asarray(
        model.obstacle_barriers(robot.default_state, field),
        dtype=np.float64,
    )
    offset = 0
    maximum_jump = 0.0
    for line_length in lines:
        line_values = values[offset : offset + line_length]
        maximum_jump = max(
            maximum_jump,
            float(np.max(np.abs(np.diff(line_values)))),
        )
        offset += line_length

    # The former hard learned/analytic domain switch produced 40-54 mm jumps
    # for a 5 mm translation. The C1 guard blend keeps this sweep continuous.
    assert maximum_jump < 0.01


def _test_state() -> jnp.ndarray:
    return jnp.asarray(
        (
            0.35,
            -0.27,
            0.41,
            0.52,
            -0.31,
            0.27,
            -0.38,
            -0.63,
            0.44,
            -0.29,
            0.21,
        ),
        dtype=jnp.float32,
    )


def _oriented_rectangle_sdf(
    point: jnp.ndarray,
    polygon: jnp.ndarray,
) -> jnp.ndarray:
    center = jnp.mean(polygon, axis=0)
    x_edge = polygon[1] - polygon[0]
    y_edge = polygon[3] - polygon[0]
    half_extents = 0.5 * jnp.asarray(
        (jnp.linalg.norm(x_edge), jnp.linalg.norm(y_edge))
    )
    axes = jnp.stack(
        (
            x_edge / jnp.linalg.norm(x_edge),
            y_edge / jnp.linalg.norm(y_edge),
        ),
        axis=1,
    )
    local = (point - center) @ axes
    q = jnp.abs(local) - half_extents
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0))
    inside = jnp.minimum(jnp.maximum(q[0], q[1]), 0.0)
    return outside + inside


def _continuous_rectangle_clearances(
    robot: MobileArmRobot,
    state: jnp.ndarray,
    field: ObstacleField,
) -> jnp.ndarray:
    polygons = jnp.concatenate(
        (
            robot.base_polygon(state)[None, :, :],
            robot.link_polygons(state),
        ),
        axis=0,
    )
    per_part = jax.vmap(
        lambda point: jax.vmap(
            lambda polygon: _oriented_rectangle_sdf(point, polygon)
        )(polygons)
    )(field.centers)
    return (
        jnp.min(per_part, axis=1)
        - field.radii
        - robot.body_point_radius
    )
