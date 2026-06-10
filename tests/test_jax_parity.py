from __future__ import annotations

import json
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import MPPIConfig, MPPIController
from robots import create_robot
from sdf import default_obstacle_field


BASELINE_FILES = (
    Path("tests/baselines/numpy_si_brmppi_deterministic.json"),
    Path("tests/baselines/numpy_uni_brmppi_deterministic.json"),
    Path("tests/baselines/numpy_du_brmppi_deterministic.json"),
    Path("tests/baselines/numpy_quad_brmppi_deterministic.json"),
    Path("tests/baselines/numpy_mobile_brmppi_deterministic.json"),
)


@pytest.mark.parametrize("baseline_path", BASELINE_FILES)
def test_deterministic_brmppi_trace_matches_numpy_baseline(baseline_path: Path) -> None:
    data = json.loads(baseline_path.read_text())
    robot = create_robot(data["robot"])
    field = default_obstacle_field(robot.name)
    config = MPPIConfig(
        horizon=data["horizon"],
        samples=data["samples"],
        dt=data["dt"],
        noise_scale=data["noise_scale"],
        alpha_noise_scale=data["alpha_noise_scale"],
        plot_samples=0,
    )
    controller = MPPIController(robot, field, algo=data["algo"], config=config, seed=data["seed"])
    state = robot.default_state.copy()
    goal = robot.default_goal.copy()

    for trace in data["traces"]:
        action, diagnostics = controller.command(state, goal)
        next_state = robot.step(state, action, data["dt"])

        np.testing.assert_allclose(np.asarray(action), trace["action"], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.asarray(next_state), trace["next_state"], rtol=1e-8, atol=1e-8)
        assert diagnostics["best_cost"] == pytest.approx(trace["cost"], rel=1e-8, abs=1e-6)
        assert diagnostics["mean_cost"] == pytest.approx(trace["mean_cost"], rel=1e-8, abs=1e-6)
        assert diagnostics["best_min_clearance"] == pytest.approx(trace["best_min_clearance"], rel=1e-8, abs=1e-8)
        assert diagnostics["sampled_min_clearance"] == pytest.approx(trace["sampled_min_clearance"], rel=1e-8, abs=1e-8)
        assert diagnostics["sampled_collision_fraction"] == pytest.approx(
            trace["sampled_collision_fraction"],
            rel=1e-8,
            abs=1e-12,
        )
        state = next_state


def test_mobile_arm_body_geometry_shapes_are_fixed() -> None:
    robot = create_robot("mobile_arm")
    assert robot.base_polygon(robot.default_state).shape == (4, 2)
    assert robot.arm_polylines(robot.default_state).shape == (2, 5, 2)
    assert robot.link_polygons(robot.default_state).shape == (8, 4, 2)
    assert robot.body_points(robot.default_state).shape == (193, 2)


def test_controller_public_state_uses_jax_arrays() -> None:
    robot = create_robot("unicycle")
    field = default_obstacle_field(robot.name)
    controller = MPPIController(
        robot,
        field,
        algo="brmppi",
        config=MPPIConfig(horizon=4, samples=5, dt=0.1, plot_samples=0),
        seed=7,
    )
    action, diagnostics = controller.command(robot.default_state, robot.default_goal)

    assert isinstance(action, jax.Array)
    assert isinstance(controller.control_sequence, jax.Array)
    assert isinstance(controller.alpha_state, jax.Array)
    assert isinstance(diagnostics["sampled_costs"], jax.Array)
    assert diagnostics["sampled_trajectories"].shape == (0, 5, robot.state_dim)
