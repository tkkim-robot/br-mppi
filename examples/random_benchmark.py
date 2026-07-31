from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig, MPPIController, load_tuned_config
from robots import ROBOT_REGISTRY, create_robot
from sdf import CircleObstacle, ObstacleField, PretrainedSDFUnavailable, load_pretrained_sdf_for_robot


ROBOT_DEFAULTS = {
    "single_integrator": {"horizon": 20, "samples": 80, "max_steps": 800, "min_obstacles": 20, "max_obstacles": 20},
    "unicycle": {"horizon": 36, "samples": 80, "max_steps": 900, "min_obstacles": 20, "max_obstacles": 20},
    "dynamic_unicycle": {"horizon": 20, "samples": 80, "max_steps": 900, "min_obstacles": 20, "max_obstacles": 20},
    "planar_quadrotor": {"horizon": 36, "samples": 80, "max_steps": 900, "min_obstacles": 20, "max_obstacles": 20},
    "mobile_arm": {"horizon": 28, "samples": 96, "max_steps": 1000, "min_obstacles": 16, "max_obstacles": 16},
}
DEFAULT_DEADLOCK_WINDOW = 80
DEFAULT_DEADLOCK_POSITION_TOLERANCE = 0.03
DEFAULT_DEADLOCK_PROGRESS_TOLERANCE = 0.02


@dataclass(frozen=True)
class TrialResult:
    trial: int
    field_seed: int
    controller_seed: int
    robot: str
    algo: str
    obstacle_count: int
    reached: bool
    collision: bool
    timeout: bool
    steps: int
    final_error: float
    min_exact_clearance: float
    min_continuous_clearance: float | None
    sampled_min_clearance: float
    best_rollout_min_clearance: float
    first_sample_collision_step: int | None
    first_best_collision_step: int | None
    max_sampled_collision_fraction: float
    mean_command_ms: float
    p95_command_ms: float
    wall_seconds: float
    deadlock: bool = False
    deadlock_window: int = 0
    deadlock_position_tolerance: float = 0.0
    deadlock_progress_tolerance: float = 0.0
    barrier_source: str = "analytic_sdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPPI-family benchmarks across randomized circular-obstacle fields.",
    )
    parser.add_argument("--algo", "--method", "--model", choices=("all", *ALGORITHMS), default="all")
    parser.add_argument("--robot", "--dynamics", choices=tuple(sorted(ROBOT_REGISTRY)), default="unicycle")
    parser.add_argument(
        "--nsdf",
        action="store_true",
        help="Use the matching pretrained neural SDF. This is supported only with --algo brmppi.",
    )
    parser.add_argument(
        "--nsdf-variant",
        choices=("legacy", "retrained"),
        default="retrained",
        help="Checkpoint family selected by --nsdf (default: retrained).",
    )
    parser.add_argument(
        "--mobile-nsdf-points",
        type=int,
        default=64,
        help="Mobile-arm obstacle-boundary stencil size for retrained NSDF inference.",
    )
    parser.add_argument(
        "--mobile-nsdf-narrow-points",
        type=int,
        default=4,
        help="Nearest mobile-arm surface samples evaluated by each learned narrow phase.",
    )
    parser.add_argument(
        "--mobile-nsdf-analytic-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conservatively cap mobile-arm learned barriers with exact broad-phase clearance.",
    )
    parser.add_argument(
        "--tuned-config",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Load repository tuned parameters. Defaults to true for a single brmppi benchmark.",
    )
    parser.add_argument("--trials", "--trails", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13, help="Base seed for randomized obstacle fields.")
    parser.add_argument("--controller-seed", type=int, default=7, help="Base seed for MPPI sampling.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Generous per-trial simulation cap.")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--min-obstacles", type=int, default=None, help="Defaults match the standard demo scene count.")
    parser.add_argument("--max-obstacles", type=int, default=None, help="Defaults match the standard demo scene count.")
    parser.add_argument("--workspace-margin", type=float, default=2.5)
    parser.add_argument("--start-clearance", type=float, default=0.35)
    parser.add_argument("--goal-clearance", type=float, default=0.9)
    parser.add_argument(
        "--deadlock-window",
        type=int,
        default=DEFAULT_DEADLOCK_WINDOW,
        help="Consecutive simulation steps used for deadlock timeout detection; 0 disables it.",
    )
    parser.add_argument(
        "--deadlock-position-tolerance",
        type=float,
        default=DEFAULT_DEADLOCK_POSITION_TOLERANCE,
        help="Maximum position span over the deadlock window before treating the trial as stuck.",
    )
    parser.add_argument(
        "--deadlock-progress-tolerance",
        type=float,
        default=DEFAULT_DEADLOCK_PROGRESS_TOLERANCE,
        help="Maximum goal-distance improvement over the deadlock window before treating the trial as stuck.",
    )
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nsdf and args.algo != "brmppi":
        raise NotImplementedError(
            f"--nsdf is only implemented for brmppi, not {args.algo}. "
            "Select --algo brmppi when benchmarking the neural SDF."
        )
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    defaults = ROBOT_DEFAULTS[args.robot]
    use_tuned_config = args.tuned_config if args.tuned_config is not None else args.algo == "brmppi"
    max_steps = defaults["max_steps"] if args.max_steps is None else args.max_steps
    min_obstacles = (
        defaults["min_obstacles"]
        if args.min_obstacles is None
        else args.min_obstacles
    )
    max_obstacles = (
        defaults["max_obstacles"]
        if args.max_obstacles is None
        else args.max_obstacles
    )
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if min_obstacles <= 0 or max_obstacles < min_obstacles:
        raise ValueError("--min-obstacles must be positive and <= --max-obstacles")
    if args.deadlock_window < 0:
        raise ValueError("--deadlock-window must be nonnegative")
    if (
        args.deadlock_position_tolerance < 0.0
        or args.deadlock_progress_tolerance < 0.0
    ):
        raise ValueError("deadlock tolerances must be nonnegative")

    algos = ALGORITHMS if args.algo == "all" else (args.algo,)
    configs = resolve_benchmark_configs(args, algos, use_tuned_config=use_tuned_config)
    sdf_model = load_benchmark_sdf(
        args.robot,
        nsdf=args.nsdf,
        variant=args.nsdf_variant,
        mobile_points=args.mobile_nsdf_points,
        mobile_narrow_points=args.mobile_nsdf_narrow_points,
        mobile_analytic_guard=args.mobile_nsdf_analytic_guard,
    )
    barrier_source = "neural_sdf" if sdf_model is not None else "analytic_sdf"
    barrier_implementation = (
        "analytic_sdf"
        if sdf_model is None
        else (
            "guarded_analytic_neural_hybrid"
            if args.robot == "mobile_arm" and args.mobile_nsdf_analytic_guard
            else "neural_sdf"
        )
    )
    run_parameters = {
        algo: {
            "horizon": configs[algo].horizon if configs[algo] is not None else (args.horizon or defaults["horizon"]),
            "samples": configs[algo].samples if configs[algo] is not None else (args.samples or defaults["samples"]),
            "dt": configs[algo].dt if configs[algo] is not None else (args.dt if args.dt is not None else 0.1),
        }
        for algo in algos
    }
    robot = create_robot(args.robot)
    results: list[TrialResult] = []
    runtime = runtime_provenance()
    benchmark_started_utc = datetime.now(timezone.utc).isoformat()

    for trial in range(args.trials):
        field_seed = args.seed + trial
        rng = np.random.default_rng(field_seed)
        field = random_obstacle_field(
            robot,
            rng,
            min_obstacles=min_obstacles,
            max_obstacles=max_obstacles,
            workspace_margin=args.workspace_margin,
            start_clearance=args.start_clearance,
            goal_clearance=args.goal_clearance,
        )
        for algo in algos:
            controller_seed = args.controller_seed + trial
            config = configs.get(algo)
            horizon = run_parameters[algo]["horizon"]
            samples = run_parameters[algo]["samples"]
            dt = run_parameters[algo]["dt"]
            result = run_trial(
                robot_name=args.robot,
                field=field,
                algo=algo,
                trial=trial,
                field_seed=field_seed,
                controller_seed=controller_seed,
                horizon=horizon,
                samples=samples,
                max_steps=max_steps,
                dt=dt,
                warmup=args.warmup,
                config=config,
                sdf_model=sdf_model,
                deadlock_window=args.deadlock_window,
                deadlock_position_tolerance=args.deadlock_position_tolerance,
                deadlock_progress_tolerance=args.deadlock_progress_tolerance,
            )
            results.append(result)
            if not args.quiet:
                print_trial_result(result)

    report = {
        "schema_version": 2,
        "settings": {
            "robot": args.robot,
            "algos": list(algos),
            "trials": args.trials,
            "barrier_source": barrier_source,
            "barrier_implementation": barrier_implementation,
            "pretrained_sdf": None if sdf_model is None else sdf_model.description,
            "nsdf_variant": args.nsdf_variant if sdf_model is not None else None,
            "nsdf_provenance": (
                None
                if sdf_model is None
                else dict(getattr(sdf_model, "provenance", {}))
            ),
            "tuned_config": use_tuned_config,
            "run_parameters": run_parameters,
            "configs": {
                algo: (
                    asdict(config)
                    if config is not None
                    else asdict(
                        MPPIConfig(
                            horizon=run_parameters[algo]["horizon"],
                            samples=run_parameters[algo]["samples"],
                            dt=run_parameters[algo]["dt"],
                            plot_samples=0,
                        )
                    )
                )
                for algo, config in configs.items()
            },
            "max_steps": max_steps,
            "seed": args.seed,
            "controller_seed": args.controller_seed,
            "min_obstacles": min_obstacles,
            "max_obstacles": max_obstacles,
            "workspace_margin": args.workspace_margin,
            "start_clearance": args.start_clearance,
            "goal_clearance": args.goal_clearance,
            "deadlock_window": args.deadlock_window,
            "deadlock_position_tolerance": args.deadlock_position_tolerance,
            "deadlock_progress_tolerance": args.deadlock_progress_tolerance,
            "warmup_excluded_from_timing": args.warmup,
            "command_timing_scope": "controller.command plus device synchronization",
            "wall_seconds_includes_benchmark_diagnostics": True,
            "benchmark_started_utc": benchmark_started_utc,
            "benchmark_finished_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": runtime,
        },
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
    if len(algos) == 1 and configs.get(algos[0]) is not None:
        report["best_config"] = asdict(configs[algos[0]])
    output_path = args.output or default_output_path(
        args.robot,
        algos,
        barrier_source=barrier_source,
        nsdf_variant=args.nsdf_variant if sdf_model is not None else None,
        tuned_config=use_tuned_config,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=json_default),
        encoding="utf-8",
    )

    print_summary(report["summary"])
    print(f"results={output_path}")


def resolve_benchmark_configs(
    args: argparse.Namespace,
    algos: tuple[str, ...],
    *,
    use_tuned_config: bool,
) -> dict[str, MPPIConfig | None]:
    if not use_tuned_config:
        return {algo: None for algo in algos}

    overrides = {"plot_samples": 0}
    if args.horizon is not None:
        overrides["horizon"] = args.horizon
    if args.samples is not None:
        overrides["samples"] = args.samples
    if args.dt is not None:
        overrides["dt"] = args.dt
    return {
        algo: load_tuned_config(args.robot, algo, overrides=overrides)
        for algo in algos
    }


def load_benchmark_sdf(
    robot_name: str,
    *,
    nsdf: bool,
    variant: str = "legacy",
    mobile_points: int = 64,
    mobile_narrow_points: int = 4,
    mobile_analytic_guard: bool = True,
):
    if not nsdf:
        return None
    try:
        model = load_pretrained_sdf_for_robot(
            robot_name,
            repo_root=REPO_ROOT,
            variant=variant,
            mobile_arm_points_per_obstacle=mobile_points,
            mobile_arm_narrow_phase_points=mobile_narrow_points,
        )
        if robot_name == "mobile_arm":
            model.analytic_guard = mobile_analytic_guard
        return model
    except PretrainedSDFUnavailable as exc:
        raise NotImplementedError(
            f"--nsdf --nsdf-variant {variant} is not implemented for {robot_name}: {exc}"
        ) from exc


def random_obstacle_field(
    robot,
    rng: np.random.Generator,
    *,
    min_obstacles: int,
    max_obstacles: int,
    workspace_margin: float,
    start_clearance: float,
    goal_clearance: float,
) -> ObstacleField:
    start = np.asarray(robot.position(robot.default_state), dtype=float)
    goal = np.asarray(robot.default_goal, dtype=float)
    low = np.minimum(start, goal) - np.array([workspace_margin, workspace_margin + 2.0])
    high = np.maximum(start, goal) + np.array([workspace_margin, workspace_margin + 2.0])
    axis = goal - start
    length = max(float(np.linalg.norm(axis)), 1e-6)
    tangent = axis / length
    normal = np.array([-tangent[1], tangent[0]])
    body_points = np.asarray(robot.body_points(robot.default_state), dtype=float)
    obstacle_count = int(rng.integers(min_obstacles, max_obstacles + 1))
    radius_low, radius_high = radius_range(robot.name)
    placement_attempts = 4000
    field_attempts = 8

    for _field_attempt in range(field_attempts):
        obstacles: list[CircleObstacle] = []
        field_failed = False
        for _ in range(obstacle_count):
            accepted = False
            for _placement_attempt in range(placement_attempts):
                radius = float(rng.uniform(radius_low, radius_high))
                if rng.random() < 0.72:
                    t = float(rng.uniform(0.08, 0.92))
                    offset = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.35, 2.4 + 0.2 * radius))
                    along_jitter = float(rng.normal(0.0, 0.45))
                    cross_jitter = float(rng.normal(0.0, 0.18))
                    center = start + t * axis + normal * (offset + cross_jitter) + tangent * along_jitter
                else:
                    center = rng.uniform(low, high)
                center = np.asarray(center, dtype=float)
                if is_valid_obstacle(
                    center,
                    radius,
                    obstacles,
                    body_points,
                    start,
                    goal,
                    low,
                    high,
                    robot.body_point_radius,
                    start_clearance,
                    goal_clearance,
                ):
                    obstacles.append(CircleObstacle(center=(float(center[0]), float(center[1])), radius=radius))
                    accepted = True
                    break
            if not accepted:
                field_failed = True
                break
        if not field_failed:
            return ObstacleField(obstacles=tuple(obstacles))

    raise RuntimeError(
        "failed to place a valid randomized obstacle field "
        f"for {robot.name}: requested {obstacle_count} obstacles after "
        f"{field_attempts} field attempts x {placement_attempts} placement attempts. "
        "Relax obstacle count, workspace margin, or clearance settings."
    )


def radius_range(robot_name: str) -> tuple[float, float]:
    if robot_name == "mobile_arm":
        return 0.28, 0.62
    if robot_name in {"unicycle", "dynamic_unicycle"}:
        return 0.38, 0.72
    return 0.34, 0.68


def is_valid_obstacle(
    center: np.ndarray,
    radius: float,
    obstacles: list[CircleObstacle],
    body_points: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    body_point_radius: float,
    start_clearance: float,
    goal_clearance: float,
) -> bool:
    if np.any(center < low) or np.any(center > high):
        return False
    start_distance = np.min(np.linalg.norm(body_points - center[None, :], axis=1)) - radius - body_point_radius
    if start_distance < start_clearance:
        return False
    if np.linalg.norm(center - start) < radius + start_clearance:
        return False
    if np.linalg.norm(center - goal) < radius + goal_clearance:
        return False
    for obstacle in obstacles:
        other_center = np.asarray(obstacle.center, dtype=float)
        min_spacing = 0.65 * (radius + obstacle.radius)
        if np.linalg.norm(center - other_center) < min_spacing:
            return False
    return True


def run_trial(
    *,
    robot_name: str,
    field: ObstacleField,
    algo: str,
    trial: int,
    field_seed: int,
    controller_seed: int,
    horizon: int,
    samples: int,
    max_steps: int,
    dt: float,
    warmup: bool,
    config: MPPIConfig | None = None,
    sdf_model=None,
    deadlock_window: int = DEFAULT_DEADLOCK_WINDOW,
    deadlock_position_tolerance: float = DEFAULT_DEADLOCK_POSITION_TOLERANCE,
    deadlock_progress_tolerance: float = DEFAULT_DEADLOCK_PROGRESS_TOLERANCE,
) -> TrialResult:
    robot = create_robot(robot_name)
    controller_config = replace(config, plot_samples=0) if config is not None else MPPIConfig(
        horizon=horizon,
        samples=samples,
        dt=dt,
        plot_samples=0,
    )
    controller = MPPIController(
        robot,
        field,
        algo=algo,
        sdf_model=sdf_model,
        config=controller_config,
        seed=controller_seed,
    )
    if warmup:
        action, _diagnostics = controller.command(robot.default_state.copy(), robot.default_goal.copy())
        jax.block_until_ready(action)
        reset_controller(controller, controller_seed)

    state = robot.default_state.copy()
    min_exact_clearance = exact_clearance(field, robot, state)
    min_continuous_clearance = continuous_geometry_clearance(field, robot, state)
    min_sampled_rollout_clearance = float("inf")
    min_best_rollout_clearance = float("inf")
    max_sampled_collision_fraction = 0.0
    first_sample_collision_step: int | None = None
    first_best_collision_step: int | None = None
    reached = False
    collision = min_exact_clearance < 0.0
    deadlock = False
    steps_run = 0
    command_times: list[float] = []
    position_history = [np.asarray(robot.position(state), dtype=float)]
    goal_distance_history = [float(jnp.linalg.norm(robot.position(state) - robot.default_goal))]
    wall_start = time.perf_counter()

    for step_idx in range(max_steps):
        steps_run = step_idx + 1
        command_start = time.perf_counter()
        action, diagnostics = controller.command(state, robot.default_goal)
        jax.block_until_ready(action)
        command_times.append(time.perf_counter() - command_start)

        sampled_min = float(diagnostics["sampled_min_clearance"])
        best_min = float(diagnostics["best_min_clearance"])
        min_sampled_rollout_clearance = min(min_sampled_rollout_clearance, sampled_min)
        min_best_rollout_clearance = min(min_best_rollout_clearance, best_min)
        max_sampled_collision_fraction = max(
            max_sampled_collision_fraction,
            float(diagnostics["sampled_collision_fraction"]),
        )
        if first_sample_collision_step is None and int(diagnostics["sampled_collision_count"]) > 0:
            first_sample_collision_step = step_idx
        if first_best_collision_step is None and bool(diagnostics["best_collision"]):
            first_best_collision_step = step_idx

        state = robot.step(state, action, dt)
        position = np.asarray(robot.position(state), dtype=float)
        goal_distance = float(jnp.linalg.norm(robot.position(state) - robot.default_goal))
        position_history.append(position)
        goal_distance_history.append(goal_distance)
        current_clearance = exact_clearance(field, robot, state)
        min_exact_clearance = min(min_exact_clearance, current_clearance)
        current_continuous_clearance = continuous_geometry_clearance(
            field,
            robot,
            state,
        )
        if current_continuous_clearance is not None:
            assert min_continuous_clearance is not None
            min_continuous_clearance = min(
                min_continuous_clearance,
                current_continuous_clearance,
            )
        if current_clearance < 0.0:
            collision = True
            break
        if goal_distance <= robot.goal_tolerance:
            reached = True
            break
        if is_deadlocked(
            position_history,
            goal_distance_history,
            deadlock_window=deadlock_window,
            position_tolerance=deadlock_position_tolerance,
            progress_tolerance=deadlock_progress_tolerance,
        ):
            deadlock = True
            break

    final_error = float(jnp.linalg.norm(robot.position(state) - robot.default_goal))
    mean_ms, p95_ms = timing_stats_ms(command_times)
    return TrialResult(
        trial=trial,
        field_seed=field_seed,
        controller_seed=controller_seed,
        robot=robot.name,
        algo=algo,
        obstacle_count=len(field.obstacles),
        reached=reached,
        collision=collision,
        timeout=not reached and not collision,
        steps=steps_run,
        final_error=final_error,
        min_exact_clearance=float(min_exact_clearance),
        min_continuous_clearance=(
            None
            if min_continuous_clearance is None
            else float(min_continuous_clearance)
        ),
        sampled_min_clearance=float(min_sampled_rollout_clearance),
        best_rollout_min_clearance=float(min_best_rollout_clearance),
        first_sample_collision_step=first_sample_collision_step,
        first_best_collision_step=first_best_collision_step,
        max_sampled_collision_fraction=float(max_sampled_collision_fraction),
        mean_command_ms=mean_ms,
        p95_command_ms=p95_ms,
        wall_seconds=time.perf_counter() - wall_start,
        deadlock=deadlock,
        deadlock_window=deadlock_window,
        deadlock_position_tolerance=deadlock_position_tolerance,
        deadlock_progress_tolerance=deadlock_progress_tolerance,
        barrier_source=getattr(controller, "barrier_source", "analytic_sdf"),
    )


def is_deadlocked(
    position_history: list[np.ndarray],
    goal_distance_history: list[float],
    *,
    deadlock_window: int,
    position_tolerance: float,
    progress_tolerance: float,
) -> bool:
    if deadlock_window <= 0 or len(position_history) <= deadlock_window:
        return False
    recent_positions = np.asarray(position_history[-(deadlock_window + 1) :], dtype=float)
    recent_goal_distances = goal_distance_history[-(deadlock_window + 1) :]
    position_span = float(np.max(np.linalg.norm(recent_positions - recent_positions[0], axis=1)))
    goal_progress = float(recent_goal_distances[0] - recent_goal_distances[-1])
    return position_span <= position_tolerance and goal_progress <= progress_tolerance


def reset_controller(controller: MPPIController, seed: int) -> None:
    controller.control_sequence = jnp.zeros_like(controller.control_sequence)
    controller.alpha_state = jnp.full((controller.num_barriers,), controller.config.barrier_alpha, dtype=float)
    controller.key = jax.random.PRNGKey(seed)


def exact_clearance(field: ObstacleField, robot, state: jnp.ndarray) -> float:
    return float(jnp.min(field.signed_distance(robot.body_points(state))) - robot.body_point_radius)


def continuous_geometry_clearance(
    field: ObstacleField,
    robot,
    state: jnp.ndarray,
) -> float | None:
    """Exact circle clearance to the mobile arm's continuous rectangle union.

    The benchmark's historical outcome metric remains ``exact_clearance`` so
    old analytic reports stay paired and comparable. This additional metric
    exposes the small geometry difference between sampled body points and the
    continuous base/link rectangles used by the compositional NSDF.
    """
    if robot.name != "mobile_arm":
        return None
    polygons = jnp.concatenate(
        (
            robot.base_polygon(state)[None, :, :],
            robot.link_polygons(state),
        ),
        axis=0,
    )
    polygon_centers = jnp.mean(polygons, axis=1)
    x_edges = polygons[:, 1] - polygons[:, 0]
    y_edges = polygons[:, 3] - polygons[:, 0]
    x_lengths = jnp.linalg.norm(x_edges, axis=1)
    y_lengths = jnp.linalg.norm(y_edges, axis=1)
    x_axes = x_edges / x_lengths[:, None]
    y_axes = y_edges / y_lengths[:, None]
    half_extents = 0.5 * jnp.stack((x_lengths, y_lengths), axis=1)

    delta = field.centers[:, None, :] - polygon_centers[None, :, :]
    local = jnp.stack(
        (
            jnp.einsum("opi,pi->op", delta, x_axes),
            jnp.einsum("opi,pi->op", delta, y_axes),
        ),
        axis=2,
    )
    offset = jnp.abs(local) - half_extents[None, :, :]
    outside = jnp.linalg.norm(jnp.maximum(offset, 0.0), axis=2)
    inside = jnp.minimum(jnp.maximum(offset[:, :, 0], offset[:, :, 1]), 0.0)
    center_to_part = outside + inside
    obstacle_clearance = (
        jnp.min(center_to_part, axis=1)
        - field.radii
        - robot.body_point_radius
    )
    return float(jnp.min(obstacle_clearance))


def timing_stats_ms(times: list[float]) -> tuple[float, float]:
    if not times:
        return 0.0, 0.0
    ordered = sorted(times)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return 1000.0 * sum(times) / len(times), 1000.0 * ordered[p95_index]


def summarize_results(results: list[TrialResult]) -> list[dict[str, object]]:
    summary = []
    for algo in sorted({result.algo for result in results}):
        rows = [result for result in results if result.algo == algo]
        count = len(rows)
        has_continuous_clearance = any(
            result.min_continuous_clearance is not None for result in rows
        )
        summary.append(
            {
                "algo": algo,
                "trials": count,
                "reached": sum(result.reached for result in rows),
                "collisions": sum(result.collision for result in rows),
                "timeouts": sum(result.timeout for result in rows),
                "deadlocks": sum(result.deadlock for result in rows),
                "success_rate": sum(result.reached for result in rows) / count,
                "collision_rate": sum(result.collision for result in rows) / count,
                "mean_steps": sum(result.steps for result in rows) / count,
                "mean_command_ms": sum(result.mean_command_ms for result in rows) / count,
                "mean_trial_p95_command_ms": (
                    sum(result.p95_command_ms for result in rows) / count
                ),
                "mean_final_error": sum(result.final_error for result in rows) / count,
                "worst_min_clearance": min(result.min_exact_clearance for result in rows),
                "continuous_collisions": (
                    sum(
                        result.min_continuous_clearance is not None
                        and result.min_continuous_clearance < 0.0
                        for result in rows
                    )
                    if has_continuous_clearance
                    else None
                ),
                "worst_continuous_clearance": (
                    min(
                        result.min_continuous_clearance
                        for result in rows
                        if result.min_continuous_clearance is not None
                    )
                    if has_continuous_clearance
                    else None
                ),
            }
        )
    return summary


def print_trial_result(result: TrialResult) -> None:
    status = "reached" if result.reached else ("collision" if result.collision else ("deadlock" if result.deadlock else "timeout"))
    continuous = (
        ""
        if result.min_continuous_clearance is None
        else f" continuous={result.min_continuous_clearance:+.3f}"
    )
    print(
        f"trial={result.trial:03d} seed={result.field_seed} "
        f"algo={result.algo:12s} robot={result.robot:18s} source={result.barrier_source:12s} status={status:9s} "
        f"steps={result.steps:4d} obs={result.obstacle_count:2d} "
        f"min_clear={result.min_exact_clearance:+.3f}{continuous} "
        f"final_error={result.final_error:.2f} mean={result.mean_command_ms:.2f}ms",
        flush=True,
    )


def print_summary(summary: list[dict[str, object]]) -> None:
    print(
        "algo | trials | reached | collisions | timeouts | deadlocks | success | "
        "collision | continuous_collisions | mean_ms | trial_p95_ms | mean_steps | "
        "worst_clear | worst_continuous"
    )
    print(
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---:"
    )
    for row in summary:
        continuous = row["worst_continuous_clearance"]
        print(
            " | ".join(
                (
                    str(row["algo"]),
                    str(row["trials"]),
                    str(row["reached"]),
                    str(row["collisions"]),
                    str(row["timeouts"]),
                    str(row["deadlocks"]),
                    f"{float(row['success_rate']):.3f}",
                    f"{float(row['collision_rate']):.3f}",
                    (
                        "n/a"
                        if row["continuous_collisions"] is None
                        else str(row["continuous_collisions"])
                    ),
                    f"{float(row['mean_command_ms']):.2f}",
                    f"{float(row['mean_trial_p95_command_ms']):.2f}",
                    f"{float(row['mean_steps']):.1f}",
                    f"{float(row['worst_min_clearance']):+.3f}",
                    (
                        "n/a"
                        if continuous is None
                        else f"{float(continuous):+.3f}"
                    ),
                )
            )
        )


def default_output_path(
    robot_name: str,
    algos: tuple[str, ...],
    *,
    barrier_source: str = "analytic_sdf",
    nsdf_variant: str | None = None,
    tuned_config: bool = False,
) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    algo_label = "all" if len(algos) > 1 else algos[0]
    config_label = "tuned" if tuned_config else "default"
    source_label = (
        f"{barrier_source}_{nsdf_variant}"
        if barrier_source == "neural_sdf" and nsdf_variant is not None
        else barrier_source
    )
    return (
        Path("output")
        / "benchmarks"
        / f"random_obstacles_{robot_name}_{algo_label}_{source_label}_{config_label}_{stamp}.json"
    )


def runtime_provenance() -> dict[str, object]:
    revision = _git_output("rev-parse", "HEAD")
    dirty_output = _git_output("status", "--porcelain")
    return {
        "git_revision": revision,
        "git_dirty": None if dirty_output is None else bool(dirty_output),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": str(np.__version__),
        "jax_version": str(jax.__version__),
        "jaxlib_version": str(jax.lib.__version__),
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [
            {
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "id": int(device.id),
            }
            for device in jax.devices()
        ],
    }


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def json_default(value):
    """Convert common provenance scalar/array types to JSON primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
