from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig, MPPIController
from robots import ROBOT_REGISTRY, create_robot
from sdf import CircleObstacle, ObstacleField


ROBOT_DEFAULTS = {
    "single_integrator": {"horizon": 20, "samples": 80, "max_steps": 800, "min_obstacles": 20, "max_obstacles": 20},
    "unicycle": {"horizon": 36, "samples": 80, "max_steps": 900, "min_obstacles": 20, "max_obstacles": 20},
    "dynamic_unicycle": {"horizon": 20, "samples": 80, "max_steps": 900, "min_obstacles": 20, "max_obstacles": 20},
    "planar_quadrotor": {"horizon": 36, "samples": 80, "max_steps": 900, "min_obstacles": 20, "max_obstacles": 20},
    "mobile_arm": {"horizon": 28, "samples": 96, "max_steps": 1000, "min_obstacles": 16, "max_obstacles": 16},
}


@dataclass(frozen=True)
class TrialResult:
    trial: int
    field_seed: int
    robot: str
    algo: str
    obstacle_count: int
    reached: bool
    collision: bool
    timeout: bool
    steps: int
    final_error: float
    min_exact_clearance: float
    sampled_min_clearance: float
    best_rollout_min_clearance: float
    first_sample_collision_step: int | None
    first_best_collision_step: int | None
    max_sampled_collision_fraction: float
    mean_command_ms: float
    p95_command_ms: float
    wall_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MPPI-family benchmarks across randomized circular-obstacle fields.",
    )
    parser.add_argument("--algo", "--model", choices=("all", *ALGORITHMS), default="all")
    parser.add_argument("--robot", "--dynamics", choices=tuple(sorted(ROBOT_REGISTRY)), default="unicycle")
    parser.add_argument("--trials", "--trails", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13, help="Base seed for randomized obstacle fields.")
    parser.add_argument("--controller-seed", type=int, default=7, help="Base seed for MPPI sampling.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Generous per-trial simulation cap.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--min-obstacles", type=int, default=None, help="Defaults match the standard demo scene count.")
    parser.add_argument("--max-obstacles", type=int, default=None, help="Defaults match the standard demo scene count.")
    parser.add_argument("--workspace-margin", type=float, default=2.5)
    parser.add_argument("--start-clearance", type=float, default=0.35)
    parser.add_argument("--goal-clearance", type=float, default=0.9)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = ROBOT_DEFAULTS[args.robot]
    horizon = args.horizon or defaults["horizon"]
    samples = args.samples or defaults["samples"]
    max_steps = args.max_steps or defaults["max_steps"]
    min_obstacles = args.min_obstacles or defaults["min_obstacles"]
    max_obstacles = args.max_obstacles or defaults["max_obstacles"]
    if min_obstacles <= 0 or max_obstacles < min_obstacles:
        raise ValueError("--min-obstacles must be positive and <= --max-obstacles")

    algos = ALGORITHMS if args.algo == "all" else (args.algo,)
    robot = create_robot(args.robot)
    results: list[TrialResult] = []

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
                dt=args.dt,
                warmup=args.warmup,
            )
            results.append(result)
            if not args.quiet:
                print_trial_result(result)

    report = {
        "settings": {
            "robot": args.robot,
            "algos": list(algos),
            "trials": args.trials,
            "horizon": horizon,
            "samples": samples,
            "max_steps": max_steps,
            "dt": args.dt,
            "seed": args.seed,
            "controller_seed": args.controller_seed,
            "min_obstacles": min_obstacles,
            "max_obstacles": max_obstacles,
            "workspace_margin": args.workspace_margin,
            "start_clearance": args.start_clearance,
            "goal_clearance": args.goal_clearance,
            "warmup_excluded_from_timing": args.warmup,
        },
        "summary": summarize_results(results),
        "results": [asdict(result) for result in results],
    }
    output_path = args.output or default_output_path(args.robot, algos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_summary(report["summary"])
    print(f"results={output_path}")


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
    obstacles: list[CircleObstacle] = []

    for _ in range(obstacle_count):
        accepted = False
        for _attempt in range(4000):
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
            center = rng.uniform(low, high)
            obstacles.append(CircleObstacle(center=(float(center[0]), float(center[1])), radius=float(radius_low)))

    return ObstacleField(obstacles=tuple(obstacles))


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
) -> TrialResult:
    robot = create_robot(robot_name)
    controller_config = replace(config, plot_samples=0) if config is not None else MPPIConfig(
        horizon=horizon,
        samples=samples,
        dt=dt,
        plot_samples=0,
    )
    controller = MPPIController(robot, field, algo=algo, config=controller_config, seed=controller_seed)
    if warmup:
        action, _diagnostics = controller.command(robot.default_state.copy(), robot.default_goal.copy())
        jax.block_until_ready(action)
        reset_controller(controller, controller_seed)

    state = robot.default_state.copy()
    min_exact_clearance = exact_clearance(field, robot, state)
    min_sampled_rollout_clearance = float("inf")
    min_best_rollout_clearance = float("inf")
    max_sampled_collision_fraction = 0.0
    first_sample_collision_step: int | None = None
    first_best_collision_step: int | None = None
    reached = False
    collision = min_exact_clearance < 0.0
    steps_run = 0
    command_times: list[float] = []
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
        current_clearance = exact_clearance(field, robot, state)
        min_exact_clearance = min(min_exact_clearance, current_clearance)
        if current_clearance < 0.0:
            collision = True
            break
        if float(jnp.linalg.norm(robot.position(state) - robot.default_goal)) <= robot.goal_tolerance:
            reached = True
            break

    final_error = float(jnp.linalg.norm(robot.position(state) - robot.default_goal))
    mean_ms, p95_ms = timing_stats_ms(command_times)
    return TrialResult(
        trial=trial,
        field_seed=field_seed,
        robot=robot.name,
        algo=algo,
        obstacle_count=len(field.obstacles),
        reached=reached,
        collision=collision,
        timeout=not reached and not collision,
        steps=steps_run,
        final_error=final_error,
        min_exact_clearance=float(min_exact_clearance),
        sampled_min_clearance=float(min_sampled_rollout_clearance),
        best_rollout_min_clearance=float(min_best_rollout_clearance),
        first_sample_collision_step=first_sample_collision_step,
        first_best_collision_step=first_best_collision_step,
        max_sampled_collision_fraction=float(max_sampled_collision_fraction),
        mean_command_ms=mean_ms,
        p95_command_ms=p95_ms,
        wall_seconds=time.perf_counter() - wall_start,
    )


def reset_controller(controller: MPPIController, seed: int) -> None:
    controller.control_sequence = jnp.zeros_like(controller.control_sequence)
    controller.alpha_state = jnp.full((controller.num_barriers,), controller.config.barrier_alpha, dtype=float)
    controller.key = jax.random.PRNGKey(seed)


def exact_clearance(field: ObstacleField, robot, state: jnp.ndarray) -> float:
    return float(jnp.min(field.signed_distance(robot.body_points(state))) - robot.body_point_radius)


def timing_stats_ms(times: list[float]) -> tuple[float, float]:
    if not times:
        return 0.0, 0.0
    ordered = sorted(times)
    p95_index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return 1000.0 * sum(times) / len(times), 1000.0 * ordered[p95_index]


def summarize_results(results: list[TrialResult]) -> list[dict[str, float | int | str]]:
    summary = []
    for algo in sorted({result.algo for result in results}):
        rows = [result for result in results if result.algo == algo]
        count = len(rows)
        summary.append(
            {
                "algo": algo,
                "trials": count,
                "reached": sum(result.reached for result in rows),
                "collisions": sum(result.collision for result in rows),
                "timeouts": sum(result.timeout for result in rows),
                "success_rate": sum(result.reached for result in rows) / count,
                "collision_rate": sum(result.collision for result in rows) / count,
                "mean_steps": sum(result.steps for result in rows) / count,
                "mean_command_ms": sum(result.mean_command_ms for result in rows) / count,
                "mean_final_error": sum(result.final_error for result in rows) / count,
                "worst_min_clearance": min(result.min_exact_clearance for result in rows),
            }
        )
    return summary


def print_trial_result(result: TrialResult) -> None:
    status = "reached" if result.reached else ("collision" if result.collision else "timeout")
    print(
        f"trial={result.trial:03d} seed={result.field_seed} "
        f"algo={result.algo:12s} robot={result.robot:18s} status={status:9s} "
        f"steps={result.steps:4d} obs={result.obstacle_count:2d} "
        f"min_clear={result.min_exact_clearance:+.3f} "
        f"final_error={result.final_error:.2f} mean={result.mean_command_ms:.2f}ms",
        flush=True,
    )


def print_summary(summary: list[dict[str, float | int | str]]) -> None:
    print("algo | trials | reached | collisions | timeouts | success | collision | mean_ms | mean_steps | worst_clear")
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for row in summary:
        print(
            " | ".join(
                (
                    str(row["algo"]),
                    str(row["trials"]),
                    str(row["reached"]),
                    str(row["collisions"]),
                    str(row["timeouts"]),
                    f"{float(row['success_rate']):.3f}",
                    f"{float(row['collision_rate']):.3f}",
                    f"{float(row['mean_command_ms']):.2f}",
                    f"{float(row['mean_steps']):.1f}",
                    f"{float(row['worst_min_clearance']):+.3f}",
                )
            )
        )


def default_output_path(robot_name: str, algos: tuple[str, ...]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    algo_label = "all" if len(algos) > 1 else algos[0]
    return Path("output") / "benchmarks" / f"random_obstacles_{robot_name}_{algo_label}_{stamp}.json"


if __name__ == "__main__":
    main()
