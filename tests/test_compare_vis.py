"""Compare five tuned MPPI-family methods in fixed, difficult environments.

Examples
--------
Open the interactive comparison GUI::

    uv run python tests/test_compare_vis.py --dynamics unicycle

Save a synchronized MP4 without opening a window::

    uv run python tests/test_compare_vis.py --dynamics planar_quadrotor \
        --save-video --headless
"""

from __future__ import annotations

# Executable visualization; exclude it from pytest collection.
__test__ = False

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import MPPIController, load_tuned_config  # noqa: E402
from examples.random_benchmark import (  # noqa: E402
    DEFAULT_DEADLOCK_POSITION_TOLERANCE,
    DEFAULT_DEADLOCK_PROGRESS_TOLERANCE,
    DEFAULT_DEADLOCK_WINDOW,
    continuous_geometry_clearance,
    exact_clearance,
    is_deadlocked,
    reset_controller,
    timing_stats_ms,
)
from robots import ROBOT_REGISTRY, create_robot  # noqa: E402
from sdf import CircleObstacle, ObstacleField  # noqa: E402


COMPARISON_METHODS = (
    "brmppi",
    "mppi_cbf",
    "shield_mppi",
    "sc_mppi",
    "gs_mppi",
)
METHOD_LABELS = {
    "brmppi": "BRMPPI",
    "mppi_cbf": "MPPI_CBF",
    "shield_mppi": "SHIELD_MPPI",
    "sc_mppi": "SC_MPPI",
    "gs_mppi": "GS_MPPI",
}
ROBOT_LABELS = {
    "single_integrator": "Single Integrator",
    "unicycle": "Unicycle",
    "dynamic_unicycle": "Dynamic Unicycle",
    "planar_quadrotor": "Planar Quadrotor",
    "mobile_arm": "Mobile Arm",
}
DEFAULT_MAX_STEPS = 600
DEFAULT_CONTROLLER_SEED = 7
DEFAULT_PLOT_SAMPLES = 24
MAIN_SAMPLE_LINEWIDTH = 1.35
COMPARISON_SAMPLE_LINEWIDTH = 1.05
MAIN_SAMPLE_ALPHA = 0.38
COMPARISON_SAMPLE_ALPHA = 0.32


@dataclass(frozen=True)
class ScenarioSpec:
    start_state: tuple[float, ...]
    goal: tuple[float, float]
    obstacles: tuple[tuple[float, float, float], ...]
    max_steps: int = DEFAULT_MAX_STEPS

    def obstacle_field(self) -> ObstacleField:
        return ObstacleField(
            obstacles=tuple(
                CircleObstacle(center=(x, y), radius=radius)
                for x, y, radius in self.obstacles
            )
        )


# Fixed obstacle fields, stored as (center_x, center_y, radius) triples.
SCENARIOS: dict[str, ScenarioSpec] = {
    "dynamic_unicycle": ScenarioSpec(
        start_state=(2.0, 5.0, 0.0, 0.0),
        goal=(16.0, 5.0),
        obstacles=(
            (8.84, 11.01, 0.87),
            (12.6, 10.99, 0.63),
            (7.54, 10.17, 0.89),
            (5.63, 10.61, 0.51),
            (11.13, 10.03, 0.84),
            (13.54, 9.65, 0.87),
            (10.36, 8.79, 0.79),
            (12.66, 7.92, 0.87),
            (10.41, 7.76, 0.41),
            (7.7, 7.81, 0.73),
            (11.73, 7.13, 0.63),
            (13.35, 6.61, 0.68),
            (9.81, 6.49, 0.95),
            (11.38, 6.35, 0.33),
            (11.81, 5.23, 0.87),
            (12.94, 5.87, 0.37),
            (6.35, 5.48, 0.56),
            (13.34, 4.84, 0.85),
            (5.63, 5.06, 0.47),
            (6.99, 4.15, 0.88),
            (8.7, 3.17, 0.87),
            (5.66, 3.94, 0.47),
            (5.7, 2.41, 0.65),
            (8.17, 2.19, 0.4),
            (12.0, 1.72, 0.52),
            (6.64, 1.2, 0.8),
            (13.34, 0.93, 0.77),
            (9.42, 0.49, 0.61),
            (11.42, 0.32, 0.67),
            (4.51, 0.28, 0.57),
            (10.28, -0.41, 0.8),
            (13.01, -0.36, 0.76),
            (5.49, -0.79, 0.97),
            (6.78, -0.34, 0.7),
            (4.09, -0.71, 0.31),
        ),
    ),
    "mobile_arm": ScenarioSpec(
        start_state=(
            2.0,
            5.0,
            0.0,
            2.0420352248333655,
            -0.5235987755982988,
            -0.2617993877991494,
            -0.2617993877991494,
            1.0995574287564276,
            0.5235987755982988,
            0.2617993877991494,
            0.2617993877991494,
        ),
        goal=(26.0, 5.0),
        obstacles=(
            # Fixed random-style field: dense enough to resemble the other
            # hero environments while remaining reproducible across methods.
            (-1.50, -3.00, 0.70),
            (4.10, 11.80, 0.58),
            (5.75, 9.65, 0.72),
            (7.10, 13.15, 0.48),
            (8.35, 10.75, 0.64),
            (6.40, 5.15, 0.72),
            (11.65, 9.10, 0.70),
            (13.25, 13.55, 0.55),
            (9.20, 5.65, 0.68),
            (16.55, 12.10, 0.68),
            (12.10, 4.55, 0.75),
            (20.10, 13.05, 0.62),
            (15.20, 5.55, 0.72),
            (18.40, 4.45, 0.78),
            (21.70, 5.60, 0.70),
            (3.85, -1.65, 0.66),
            (5.40, 1.15, 0.48),
            (7.25, -2.45, 0.74),
            (8.80, 2.35, 0.52),
            (10.55, -0.55, 0.60),
            (12.10, 1.45, 0.43),
            (13.75, -2.05, 0.68),
            (15.30, 2.20, 0.50),
            (17.05, -0.70, 0.56),
            (18.70, 1.30, 0.72),
            (20.45, -2.30, 0.46),
            (22.30, 2.15, 0.64),
            (24.20, -0.45, 0.54),
            (27.00, 1.10, 0.70),
            (5.20, 7.65, 0.34),
            (6.55, 3.45, 0.46),
            (8.15, 6.85, 0.52),
            (9.70, 3.10, 0.38),
            (11.30, 7.35, 0.45),
            (12.85, 4.05, 0.50),
            (14.45, 6.65, 0.36),
            (16.05, 3.35, 0.48),
            (17.65, 7.20, 0.54),
            (19.30, 3.75, 0.40),
            (20.85, 6.55, 0.48),
            (22.55, 3.30, 0.44),
            (24.05, 7.10, 0.38),
        ),
        max_steps=800,
    ),
    "single_integrator": ScenarioSpec(
        start_state=(2.0, 5.0),
        goal=(22.0, 5.0),
        obstacles=(
            # Seed-83 fixed random field. Keeping the sampled coordinates in
            # source makes every controller see exactly the same environment.
            (8.81, -2.29, 0.56),
            (8.65, 2.48, 0.39),
            (18.75, 3.51, 0.44),
            (13.67, 4.84, 0.53),
            (15.39, 3.39, 0.51),
            (10.70, 6.82, 0.67),
            (12.79, 12.35, 0.62),
            (6.02, 3.85, 0.29),
            (7.85, 4.43, 0.42),
            (15.57, 5.60, 0.39),
            (12.36, 9.21, 0.62),
            (5.71, 10.58, 0.30),
            (4.19, 7.64, 0.29),
            (6.60, 10.27, 0.69),
            (15.89, 7.95, 0.53),
            (8.51, 0.83, 0.37),
            (18.50, -1.24, 0.44),
            (12.37, 5.97, 0.59),
            (11.18, 6.12, 0.64),
            (9.54, 12.22, 0.61),
            (6.09, 7.40, 0.38),
            (13.23, -2.44, 0.44),
            (19.61, 12.04, 0.36),
            (15.92, 2.44, 0.34),
            (7.64, 8.07, 0.37),
            (7.86, -3.00, 0.33),
            (9.33, 7.67, 0.71),
            (12.47, 0.29, 0.64),
            (3.44, 6.93, 0.32),
            (19.74, 12.55, 0.47),
            (10.42, -2.53, 0.52),
            (10.53, 11.84, 0.38),
            (11.51, 0.75, 0.67),
            (5.17, 0.80, 0.52),
            (17.40, 8.47, 0.50),
            (10.93, 2.31, 0.50),
            (10.06, 2.56, 0.57),
            (8.11, 1.42, 0.29),
            (17.11, 11.20, 0.29),
            (9.03, 10.64, 0.53),
            (11.84, 2.65, 0.60),
            (5.06, -2.20, 0.59),
            (9.55, 11.40, 0.46),
            (8.93, 0.41, 0.38),
            (10.86, 0.08, 0.63),
            (4.82, 11.65, 0.29),
            (16.12, -0.79, 0.41),
            (16.88, -0.20, 0.53),
        ),
    ),
    "planar_quadrotor": ScenarioSpec(
        start_state=(2.0, 5.0, 0.0, 0.0, 0.0),
        goal=(16.0, 5.0),
        obstacles=(
            (7.13, 11.46, 0.77),
            (5.46, 10.75, 0.82),
            (12.79, 10.23, 0.82),
            (7.75, 9.5, 1.14),
            (5.62, 8.96, 0.98),
            (12.92, 9.18, 1.0),
            (9.87, 8.21, 0.92),
            (4.2, 7.64, 0.65),
            (14.0, 8.32, 0.48),
            (8.29, 7.03, 1.05),
            (11.51, 5.14, 1.03),
            (6.35, 5.56, 0.66),
            (13.39, 4.06, 1.16),
            (10.08, 3.84, 0.88),
            (5.45, 4.19, 0.99),
            (11.3, 4.05, 0.83),
            (5.96, 4.5, 1.09),
            (9.07, 2.88, 0.79),
            (10.18, 3.18, 0.25),
            (10.82, 2.64, 0.51),
            (4.24, 0.93, 1.08),
            (5.61, 1.24, 0.56),
            (10.3, 2.25, 0.25),
            (11.8, 1.76, 0.72),
            (8.04, 1.67, 0.53),
            (10.29, 0.72, 0.87),
            (7.24, 0.62, 0.88),
            (8.96, 0.63, 0.86),
            (6.03, 0.2, 0.57),
            (12.18, -0.43, 0.93),
        ),
    ),
    "unicycle": ScenarioSpec(
        start_state=(2.0, 5.0, 0.0),
        goal=(16.0, 5.0),
        obstacles=(
            (6.14, 8.92, 0.62),
            (10.54, 8.63, 0.68),
            (13.73, 8.76, 0.36),
            (8.24, 8.45, 0.44),
            (8.92, 8.34, 0.34),
            (9.62, 7.91, 0.21),
            (10.71, 7.64, 0.46),
            (6.58, 7.16, 0.54),
            (11.65, 7.12, 0.69),
            (9.71, 6.98, 0.64),
            (4.91, 6.91, 0.56),
            (13.52, 7.07, 0.36),
            (12.94, 6.77, 0.38),
            (5.77, 6.55, 0.59),
            (6.47, 5.8, 0.69),
            (11.66, 5.04, 0.66),
            (4.41, 4.56, 0.72),
            (8.07, 4.77, 0.35),
            (11.26, 4.03, 0.48),
            (13.57, 3.84, 0.58),
            (8.85, 2.97, 0.56),
            (8.05, 3.36, 0.25),
            (6.89, 3.24, 0.35),
            (7.54, 3.18, 0.25),
            (12.61, 2.89, 0.67),
            (10.87, 2.43, 0.7),
            (13.61, 2.15, 0.66),
            (7.12, 2.55, 0.26),
            (8.14, 2.59, 0.1),
            (8.07, 1.67, 0.74),
            (12.5, 1.7, 0.36),
            (13.77, 1.15, 0.63),
            (9.63, 1.19, 0.46),
        ),
    ),
}


@dataclass(frozen=True)
class MethodSummary:
    dynamics: str
    method: str
    status: str
    reached: bool
    collision: bool
    deadlock: bool
    timeout: bool
    steps: int
    final_error: float
    min_clearance: float
    min_continuous_clearance: float | None
    mean_command_ms: float
    p95_command_ms: float
    wall_seconds: float
    controller_seed: int
    plot_samples: int
    tuned_config: dict[str, Any]


@dataclass
class MethodRun:
    summary: MethodSummary
    trajectory: np.ndarray
    sampled_rollouts: list[np.ndarray]
    best_rollouts: list[np.ndarray]
    clearance_history: np.ndarray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize BR-MPPI against four tuned safety baselines in "
            "fixed difficult obstacle fields."
        )
    )
    parser.add_argument(
        "--dynamics",
        "--robot",
        choices=tuple(SCENARIOS),
        default="unicycle",
        help="Robot dynamics and matching fixed obstacle configuration.",
    )
    parser.add_argument(
        "--max-steps",
        "--steps",
        type=int,
        default=None,
        help=(
            "Maximum simulation steps. By default, use the scenario limit for "
            "the selected dynamics."
        ),
    )
    parser.add_argument(
        "--controller-seed",
        "--seed",
        type=int,
        default=DEFAULT_CONTROLLER_SEED,
        help="MPPI sampling seed (default: 7).",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=DEFAULT_PLOT_SAMPLES,
        help=(
            "Sampled rollouts retained per command for visualization only "
            f"(default: {DEFAULT_PLOT_SAMPLES}; tuned control parameters are unchanged)."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--save-video",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Save an MP4. If PATH is omitted, use "
            "output/compare_vis/<dynamics>_comparison.mp4."
        ),
    )
    parser.add_argument(
        "--save-figure",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="Save the final comparison PNG, optionally at PATH.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the interactive comparison window.",
    )
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.plot_samples <= 0:
        parser.error("--plot-samples must be positive for this visualization")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.hold_seconds < 0.0:
        parser.error("--hold-seconds must be nonnegative")
    return args


def validate_scenario(dynamics: str, scenario: ScenarioSpec) -> None:
    if dynamics not in ROBOT_REGISTRY:
        raise ValueError(f"Unknown dynamics {dynamics!r}")
    robot = create_robot(dynamics)
    if len(scenario.start_state) != robot.state_dim:
        raise ValueError(
            f"{dynamics} scenario has {len(scenario.start_state)} state values; "
            f"expected {robot.state_dim}"
        )
    if len(scenario.goal) != 2:
        raise ValueError(f"{dynamics} scenario goal must have two coordinates")
    if not scenario.obstacles:
        raise ValueError(f"{dynamics} scenario must contain obstacles")
    if scenario.max_steps <= 0:
        raise ValueError(f"{dynamics} scenario max_steps must be positive")
    for index, (x, y, radius) in enumerate(scenario.obstacles):
        if not np.isfinite((x, y, radius)).all() or radius <= 0.0:
            raise ValueError(f"Invalid obstacle {index} in {dynamics} scenario")


def run_method(
    dynamics: str,
    method: str,
    *,
    max_steps: int,
    controller_seed: int,
    plot_samples: int,
) -> MethodRun:
    """Run one method with its committed tuned configuration.

    The only configuration replacement is ``plot_samples``, which controls
    diagnostics retained for drawing and does not affect controller decisions.
    """

    if method not in COMPARISON_METHODS:
        raise ValueError(f"Unsupported comparison method {method!r}")
    scenario = SCENARIOS[dynamics]
    validate_scenario(dynamics, scenario)
    robot = create_robot(dynamics)
    field = scenario.obstacle_field()
    tuned_config = load_tuned_config(dynamics, method)
    visual_config = replace(tuned_config, plot_samples=plot_samples)
    controller = MPPIController(
        robot,
        field,
        algo=method,
        config=visual_config,
        seed=controller_seed,
    )

    state = jnp.asarray(scenario.start_state, dtype=float)
    goal = jnp.asarray(scenario.goal, dtype=float)

    # Compile on the exact scenario, then reset all controller state so warm-up
    # cannot change the deterministic run. Compilation is excluded from timing.
    warmup_action, _ = controller.command(state, goal)
    jax.block_until_ready(warmup_action)
    reset_controller(controller, controller_seed)
    controller.last_diagnostics = {}

    trajectory = [np.asarray(state, dtype=float)]
    sampled_rollouts: list[np.ndarray] = []
    best_rollouts: list[np.ndarray] = []
    command_times: list[float] = []
    clearance = exact_clearance(field, robot, state)
    continuous_clearance = continuous_geometry_clearance(field, robot, state)
    clearance_history = [clearance]
    min_clearance = clearance
    min_continuous = continuous_clearance
    collision = clearance < 0.0 or (
        continuous_clearance is not None and continuous_clearance < 0.0
    )
    reached = False
    deadlock = False
    position_history = [np.asarray(robot.position(state), dtype=float)]
    goal_distance_history = [float(jnp.linalg.norm(robot.position(state) - goal))]

    wall_start = time.perf_counter()
    if not collision:
        for _step_index in range(max_steps):
            command_start = time.perf_counter()
            action, diagnostics = controller.command(state, goal)
            jax.block_until_ready(action)
            command_times.append(time.perf_counter() - command_start)

            # Move visualization arrays to host memory immediately. Otherwise a
            # long GUI run retains hundreds of device arrays.
            sampled_rollouts.append(
                np.asarray(diagnostics["sampled_trajectories"], dtype=np.float32)
            )
            best_rollouts.append(
                np.asarray(diagnostics["best_trajectory"], dtype=np.float32)
            )

            state = robot.step(state, action, visual_config.dt)
            jax.block_until_ready(state)
            trajectory.append(np.asarray(state, dtype=float))
            position = np.asarray(robot.position(state), dtype=float)
            goal_distance = float(jnp.linalg.norm(robot.position(state) - goal))
            position_history.append(position)
            goal_distance_history.append(goal_distance)

            clearance = exact_clearance(field, robot, state)
            clearance_history.append(clearance)
            min_clearance = min(min_clearance, clearance)
            current_continuous = continuous_geometry_clearance(field, robot, state)
            if current_continuous is not None:
                min_continuous = (
                    current_continuous
                    if min_continuous is None
                    else min(min_continuous, current_continuous)
                )
            if clearance < 0.0 or (
                current_continuous is not None and current_continuous < 0.0
            ):
                collision = True
                break
            if goal_distance <= robot.goal_tolerance:
                reached = True
                break
            if is_deadlocked(
                position_history,
                goal_distance_history,
                deadlock_window=DEFAULT_DEADLOCK_WINDOW,
                position_tolerance=DEFAULT_DEADLOCK_POSITION_TOLERANCE,
                progress_tolerance=DEFAULT_DEADLOCK_PROGRESS_TOLERANCE,
            ):
                deadlock = True
                break

    wall_seconds = time.perf_counter() - wall_start
    mean_ms, p95_ms = timing_stats_ms(command_times)
    final_error = float(jnp.linalg.norm(robot.position(state) - goal))
    steps = len(trajectory) - 1
    if reached:
        status = "Success"
    elif collision:
        status = "Collision"
    elif deadlock:
        status = "Deadlock"
    else:
        status = "Timeout"
    summary = MethodSummary(
        dynamics=dynamics,
        method=method,
        status=status,
        reached=reached,
        collision=collision,
        deadlock=deadlock,
        timeout=not reached and not collision and not deadlock,
        steps=steps,
        final_error=final_error,
        min_clearance=float(min_clearance),
        min_continuous_clearance=(
            None if min_continuous is None else float(min_continuous)
        ),
        mean_command_ms=mean_ms,
        p95_command_ms=p95_ms,
        wall_seconds=wall_seconds,
        controller_seed=controller_seed,
        plot_samples=plot_samples,
        tuned_config=asdict(tuned_config),
    )
    return MethodRun(
        summary=summary,
        trajectory=np.asarray(trajectory, dtype=float),
        sampled_rollouts=sampled_rollouts,
        best_rollouts=best_rollouts,
        clearance_history=np.asarray(clearance_history, dtype=float),
    )


def run_comparison(
    dynamics: str,
    *,
    max_steps: int,
    controller_seed: int,
    plot_samples: int,
) -> list[MethodRun]:
    scenario = SCENARIOS[dynamics]
    print(
        f"dynamics={dynamics} obstacles={len(scenario.obstacles)} "
        f"controller_seed={controller_seed} plot_samples={plot_samples}"
    )
    runs = []
    for method in COMPARISON_METHODS:
        print(f"running {METHOD_LABELS[method]} with tuned parameters...", flush=True)
        run = run_method(
            dynamics,
            method,
            max_steps=max_steps,
            controller_seed=controller_seed,
            plot_samples=plot_samples,
        )
        runs.append(run)
        summary = run.summary
        continuous = (
            ""
            if summary.min_continuous_clearance is None
            else f" continuous={summary.min_continuous_clearance:+.3f}"
        )
        print(
            f"  {summary.status}: steps={summary.steps} "
            f"final_error={summary.final_error:.3f} "
            f"clearance={summary.min_clearance:+.3f}{continuous} "
            f"command={summary.mean_command_ms:.2f} ms",
            flush=True,
        )
    return runs


def _configure_matplotlib(headless: bool) -> None:
    import matplotlib

    if headless:
        matplotlib.use("Agg", force=True)


def _comparison_axes(plt):
    fig = plt.figure(figsize=(20, 10), layout="constrained")
    grid = fig.add_gridspec(2, 6, width_ratios=(1, 1, 1, 1, 1, 1))
    axes = [fig.add_subplot(grid[:, :4])]
    axes.extend(
        (
            fig.add_subplot(grid[0, 4]),
            fig.add_subplot(grid[0, 5]),
            fig.add_subplot(grid[1, 4]),
            fig.add_subplot(grid[1, 5]),
        )
    )
    return fig, axes


def _axis_bounds(scenario: ScenarioSpec) -> tuple[float, float, float, float]:
    obstacles = np.asarray(scenario.obstacles, dtype=float)
    centers = obstacles[:, :2]
    radii = obstacles[:, 2]
    key_points = np.asarray((scenario.start_state[:2], scenario.goal), dtype=float)
    x_min = min(np.min(centers[:, 0] - radii), np.min(key_points[:, 0]))
    x_max = max(np.max(centers[:, 0] + radii), np.max(key_points[:, 0]))
    y_min = min(np.min(centers[:, 1] - radii), np.min(key_points[:, 1]))
    y_max = max(np.max(centers[:, 1] + radii), np.max(key_points[:, 1]))
    padding = 0.65
    return (
        float(x_min - padding),
        float(x_max + padding),
        float(y_min - padding),
        float(y_max + padding),
    )


def _draw_obstacles(ax, scenario: ScenarioSpec) -> None:
    from matplotlib.patches import Circle

    for x, y, radius in scenario.obstacles:
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="#94a3b8",
                edgecolor="#64748b",
                alpha=0.62,
                linewidth=1.2,
                zorder=4,
            )
        )


def _draw_panel(
    ax,
    dynamics: str,
    run: MethodRun,
    frame_index: int,
    *,
    is_main: bool,
) -> None:
    ax.clear()
    scenario = SCENARIOS[dynamics]
    robot = create_robot(dynamics)
    state_index = min(frame_index, len(run.trajectory) - 1)
    rollout_index = min(state_index, max(0, len(run.best_rollouts) - 1))
    _draw_obstacles(ax, scenario)

    if run.sampled_rollouts:
        sample_linewidth = (
            MAIN_SAMPLE_LINEWIDTH if is_main else COMPARISON_SAMPLE_LINEWIDTH
        )
        sample_alpha = MAIN_SAMPLE_ALPHA if is_main else COMPARISON_SAMPLE_ALPHA
        for rollout in run.sampled_rollouts[rollout_index]:
            ax.plot(
                rollout[:, 0],
                rollout[:, 1],
                color="#0d9488",
                linewidth=sample_linewidth,
                alpha=sample_alpha,
                zorder=1,
            )
    if run.best_rollouts:
        best = run.best_rollouts[rollout_index]
        ax.plot(
            best[:, 0],
            best[:, 1],
            color="#e85d75",
            linestyle="--",
            linewidth=1.7 if is_main else 1.1,
            alpha=0.75,
            zorder=2,
        )

    positions = run.trajectory[: state_index + 1, :2]
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color="#e85d75",
        linewidth=3.4 if is_main else 2.2,
        alpha=0.98,
        zorder=6,
    )
    start = np.asarray(scenario.start_state[:2], dtype=float)
    goal = np.asarray(scenario.goal, dtype=float)
    ax.scatter(
        [start[0]],
        [start[1]],
        color="#2db47d",
        edgecolor="white",
        linewidth=0.5,
        s=75 if is_main else 38,
        zorder=8,
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        marker="*",
        color="#facc15",
        edgecolor="#ca8a04",
        linewidth=0.8,
        s=230 if is_main else 115,
        zorder=8,
    )
    robot.draw(
        ax,
        run.trajectory[state_index],
        color="#4ea8de",
        edgecolor="#1d4ed8",
        zorder=9,
    )
    if run.summary.collision and state_index == len(run.trajectory) - 1:
        ax.scatter(
            [positions[-1, 0]],
            [positions[-1, 1]],
            marker="x",
            color="#dc2626",
            linewidth=2.2,
            s=90 if is_main else 45,
            zorder=12,
        )

    current_min_clearance = float(np.min(run.clearance_history[: state_index + 1]))
    terminal = state_index == len(run.trajectory) - 1
    status = f" | {run.summary.status}" if terminal else ""
    method_label = METHOD_LABELS[run.summary.method]
    if is_main:
        title = (
            f"{ROBOT_LABELS[dynamics]} | {method_label}\n"
            f"Step: {state_index}/{run.summary.steps} | "
            f"Min clear: {current_min_clearance:+.3f}{status}"
        )
        ax.set_title(title, fontsize=23, fontweight="bold", pad=10)
    else:
        status_line = f"\n{run.summary.status}" if terminal else ""
        title = (
            f"{method_label}\n{state_index}/{run.summary.steps} | "
            f"clear {current_min_clearance:+.3f}{status_line}"
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=5)

    x_min, x_max, y_min, y_max = _axis_bounds(scenario)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.15, color="#64748b", linestyle="--", zorder=0)
    ax.set_axisbelow(True)


def _draw_comparison_frame(
    axes,
    dynamics: str,
    runs: list[MethodRun],
    frame_index: int,
) -> None:
    for index, (ax, run) in enumerate(zip(axes, runs, strict=True)):
        _draw_panel(ax, dynamics, run, frame_index, is_main=index == 0)


def _frame_indices(runs: list[MethodRun], stride: int) -> list[int]:
    final_index = max(len(run.trajectory) - 1 for run in runs)
    indices = list(range(0, final_index + 1, stride))
    if not indices or indices[-1] != final_index:
        indices.append(final_index)
    return indices


def save_figure(path: Path, dynamics: str, runs: list[MethodRun]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = _comparison_axes(plt)
    final_index = max(len(run.trajectory) - 1 for run in runs)
    _draw_comparison_frame(axes, dynamics, runs, final_index)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_video(
    path: Path,
    dynamics: str,
    runs: list[MethodRun],
    *,
    frame_stride: int,
    fps: int,
    hold_seconds: float,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required for --save-video but was not found on PATH"
        )
    matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = _comparison_axes(plt)
    indices = _frame_indices(runs, frame_stride)
    hold_frames = int(round(hold_seconds * fps))
    writer = FFMpegWriter(
        fps=fps,
        codec="h264",
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"title": f"{ROBOT_LABELS[dynamics]} method comparison"},
    )
    with writer.saving(fig, str(path), dpi=120):
        for frame_index in indices:
            _draw_comparison_frame(axes, dynamics, runs, frame_index)
            writer.grab_frame()
        for _ in range(hold_frames):
            _draw_comparison_frame(axes, dynamics, runs, indices[-1])
            writer.grab_frame()
    plt.close(fig)


def show_gui(
    dynamics: str,
    runs: list[MethodRun],
    *,
    frame_stride: int,
    fps: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, axes = _comparison_axes(plt)
    indices = _frame_indices(runs, frame_stride)

    def update(frame_index: int):
        _draw_comparison_frame(axes, dynamics, runs, frame_index)
        return tuple(axes)

    animation = FuncAnimation(
        fig,
        update,
        frames=indices,
        interval=1000.0 / fps,
        repeat=False,
        blit=False,
    )
    plt.show()
    _ = animation
    plt.close(fig)


def _resolved_output_path(value: str | None, dynamics: str, suffix: str) -> Path | None:
    if value is None:
        return None
    if value == "auto":
        return Path("output") / "compare_vis" / f"{dynamics}_comparison.{suffix}"
    return Path(value)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_matplotlib(args.headless)
    max_steps = (
        SCENARIOS[args.dynamics].max_steps if args.max_steps is None else args.max_steps
    )
    runs = run_comparison(
        args.dynamics,
        max_steps=max_steps,
        controller_seed=args.controller_seed,
        plot_samples=args.plot_samples,
    )
    figure_path = _resolved_output_path(args.save_figure, args.dynamics, "png")
    video_path = _resolved_output_path(args.save_video, args.dynamics, "mp4")
    if figure_path is not None:
        save_figure(figure_path, args.dynamics, runs)
        print(f"figure={figure_path.resolve()}")
    if video_path is not None:
        save_video(
            video_path,
            args.dynamics,
            runs,
            frame_stride=args.frame_stride,
            fps=args.fps,
            hold_seconds=args.hold_seconds,
        )
        print(f"video={video_path.resolve()}")
    if not args.headless:
        show_gui(
            args.dynamics,
            runs,
            frame_stride=args.frame_stride,
            fps=args.fps,
        )
    brmppi = runs[0].summary
    if not brmppi.reached:
        print(
            f"warning: BRMPPI did not reach the goal for {args.dynamics}: "
            f"{brmppi.status}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
