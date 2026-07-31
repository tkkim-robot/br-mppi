from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (REPO_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from robots import create_robot
from sdf import CircleObstacle, ObstacleField

import random_benchmark as rb


def test_deadlock_detector_requires_position_and_progress_stall() -> None:
    stuck_positions = [np.array([0.0, 0.0]), np.array([0.002, -0.001]), np.array([0.001, 0.001])]
    stuck_goal_distances = [10.0, 9.999, 9.9985]
    assert rb.is_deadlocked(
        stuck_positions,
        stuck_goal_distances,
        deadlock_window=2,
        position_tolerance=0.01,
        progress_tolerance=0.01,
    )

    moving_positions = [np.array([0.0, 0.0]), np.array([0.02, 0.0]), np.array([0.04, 0.0])]
    assert not rb.is_deadlocked(
        moving_positions,
        stuck_goal_distances,
        deadlock_window=2,
        position_tolerance=0.01,
        progress_tolerance=0.01,
    )

    progressing_distances = [10.0, 9.95, 9.90]
    assert not rb.is_deadlocked(
        stuck_positions,
        progressing_distances,
        deadlock_window=2,
        position_tolerance=0.01,
        progress_tolerance=0.01,
    )


def test_run_trial_marks_stationary_controller_deadlock_as_timeout(monkeypatch) -> None:
    class StationaryController:
        def __init__(self, robot, *_args, **_kwargs) -> None:
            self.robot = robot

        def command(self, _state, _goal):
            return jnp.zeros((self.robot.control_dim,), dtype=float), diagnostic_payload()

    monkeypatch.setattr(rb, "MPPIController", StationaryController)
    result = rb.run_trial(
        robot_name="single_integrator",
        field=far_obstacle_field(),
        algo="brmppi",
        trial=0,
        field_seed=13,
        controller_seed=7,
        horizon=2,
        samples=3,
        max_steps=20,
        dt=0.1,
        warmup=False,
        deadlock_window=3,
        deadlock_position_tolerance=1e-9,
        deadlock_progress_tolerance=1e-9,
    )

    assert result.deadlock
    assert result.timeout
    assert not result.reached
    assert not result.collision
    assert result.steps == 3
    assert result.controller_seed == 7
    assert result.min_continuous_clearance is None
    summary = rb.summarize_results([result])[0]
    assert summary["continuous_collisions"] is None
    assert summary["worst_continuous_clearance"] is None


def test_run_trial_does_not_deadlock_when_position_moves(monkeypatch) -> None:
    class MovingController:
        def __init__(self, robot, *_args, **_kwargs) -> None:
            self.robot = robot

        def command(self, _state, _goal):
            return jnp.array([0.5, 0.0], dtype=float), diagnostic_payload()

    monkeypatch.setattr(rb, "MPPIController", MovingController)
    result = rb.run_trial(
        robot_name="single_integrator",
        field=far_obstacle_field(),
        algo="brmppi",
        trial=0,
        field_seed=13,
        controller_seed=7,
        horizon=2,
        samples=3,
        max_steps=6,
        dt=0.1,
        warmup=False,
        deadlock_window=3,
        deadlock_position_tolerance=0.01,
        deadlock_progress_tolerance=1e-9,
    )

    assert not result.deadlock
    assert result.timeout
    assert result.steps == 6


def test_run_trial_excludes_warmup_from_command_timing(monkeypatch) -> None:
    clock = {"seconds": 0.0}

    class TimedController:
        def __init__(self, robot, field, *_args, config, **_kwargs) -> None:
            self.robot = robot
            self.config = config
            self.num_barriers = len(field.obstacles)
            self.control_sequence = jnp.zeros((config.horizon, robot.control_dim + self.num_barriers))
            self.alpha_state = jnp.zeros((self.num_barriers,))
            self.calls = 0
            self.barrier_source = "analytic_sdf"

        def command(self, _state, _goal):
            self.calls += 1
            clock["seconds"] += 100.0 if self.calls == 1 else 2.0
            return jnp.zeros((self.robot.control_dim,), dtype=float), diagnostic_payload()

    monkeypatch.setattr(rb, "MPPIController", TimedController)
    monkeypatch.setattr(rb.time, "perf_counter", lambda: clock["seconds"])
    result = rb.run_trial(
        robot_name="single_integrator",
        field=far_obstacle_field(),
        algo="brmppi",
        trial=0,
        field_seed=13,
        controller_seed=7,
        horizon=2,
        samples=3,
        max_steps=1,
        dt=0.1,
        warmup=True,
    )

    assert result.mean_command_ms == 2000.0
    assert result.p95_command_ms == 2000.0
    assert result.wall_seconds == 2.0


def test_mobile_continuous_clearance_uses_full_rectangle_union() -> None:
    robot = create_robot("mobile_arm")
    state = robot.default_state
    center = robot.position(state)
    field = ObstacleField(
        obstacles=(
            CircleObstacle(
                center=(float(center[0]), float(center[1])),
                radius=0.1,
            ),
        )
    )

    clearance = rb.continuous_geometry_clearance(field, robot, state)

    assert clearance == pytest.approx(-0.62, abs=1e-6)


def far_obstacle_field() -> ObstacleField:
    return ObstacleField(obstacles=(CircleObstacle(center=(100.0, 100.0), radius=1.0),))


def diagnostic_payload() -> dict[str, float | int | bool]:
    return {
        "sampled_min_clearance": 1.0,
        "best_min_clearance": 1.0,
        "sampled_collision_fraction": 0.0,
        "sampled_collision_count": 0,
        "best_collision": False,
    }
