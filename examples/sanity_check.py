from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import MPPIConfig, MPPIController
from robots import ROBOT_REGISTRY, create_robot
from sdf import PretrainedSDFUnavailable, default_obstacle_field, load_pretrained_sdf_for_robot


@dataclass(frozen=True)
class SanityResult:
    robot: str
    algo: str
    nsdf: bool
    barrier_source: str
    reached: bool
    collision: bool
    steps: int
    final_error: float
    min_exact_clearance: float
    sampled_min_clearance: float
    best_rollout_min_clearance: float
    first_sample_collision_step: int | None
    first_best_collision_step: int | None
    max_sampled_collision_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact MPPI/BR-MPPI safety sanity checks.")
    parser.add_argument("--algo", choices=("brmppi", "mppi", "penalty_mppi"), default="brmppi")
    parser.add_argument("--robot", choices=tuple(sorted(ROBOT_REGISTRY)), default="unicycle")
    parser.add_argument("--include-nsdf", action="store_true", help="Also run the pretrained neural-SDF barrier case.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    return parser.parse_args()


def run_case(args: argparse.Namespace, *, nsdf: bool) -> SanityResult:
    robot = create_robot(args.robot)
    field = default_obstacle_field(robot.name)
    sdf_model = None
    if nsdf:
        sdf_model = load_pretrained_sdf_for_robot(robot.name, repo_root=REPO_ROOT)

    config = MPPIConfig(
        horizon=args.horizon,
        samples=args.samples,
        dt=args.dt,
        plot_samples=0,
    )
    controller = MPPIController(robot, field, algo=args.algo, sdf_model=sdf_model, config=config, seed=args.seed)
    state = robot.default_state.copy()
    goal = robot.default_goal.copy()

    min_exact_clearance = exact_clearance(field, robot, state)
    min_sampled_rollout_clearance = np.inf
    min_best_rollout_clearance = np.inf
    max_sampled_collision_fraction = 0.0
    first_sample_collision_step: int | None = None
    first_best_collision_step: int | None = None
    reached = False
    collision = min_exact_clearance < 0.0
    steps_run = 0

    for step_idx in range(args.steps):
        steps_run = step_idx + 1
        action, diagnostics = controller.command(state, goal)
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

        state = robot.step(state, action, args.dt)
        current_clearance = exact_clearance(field, robot, state)
        min_exact_clearance = min(min_exact_clearance, current_clearance)
        if current_clearance < 0.0:
            collision = True
            break
        if np.linalg.norm(robot.position(state) - goal) <= robot.goal_tolerance:
            reached = True
            break

    final_error = float(np.linalg.norm(robot.position(state) - goal))
    return SanityResult(
        robot=robot.name,
        algo=args.algo,
        nsdf=nsdf,
        barrier_source=controller.barrier_source,
        reached=reached,
        collision=collision,
        steps=steps_run,
        final_error=final_error,
        min_exact_clearance=float(min_exact_clearance),
        sampled_min_clearance=float(min_sampled_rollout_clearance),
        best_rollout_min_clearance=float(min_best_rollout_clearance),
        first_sample_collision_step=first_sample_collision_step,
        first_best_collision_step=first_best_collision_step,
        max_sampled_collision_fraction=float(max_sampled_collision_fraction),
    )


def exact_clearance(field, robot, state: np.ndarray) -> float:
    return float(np.min(field.signed_distance(robot.body_points(state))) - robot.body_point_radius)


def print_table(results: list[SanityResult]) -> None:
    header = (
        "case",
        "actual",
        "exact_min",
        "sample_min",
        "best_min",
        "first_sample",
        "first_best",
        "max_sample_frac",
    )
    print(" | ".join(header))
    print(" | ".join("---" for _ in header))
    for result in results:
        case = f"{result.algo}/{result.robot}/{result.barrier_source}"
        actual = "collision" if result.collision else "safe"
        print(
            " | ".join(
                (
                    case,
                    actual,
                    f"{result.min_exact_clearance:+.3f}",
                    f"{result.sampled_min_clearance:+.3f}",
                    f"{result.best_rollout_min_clearance:+.3f}",
                    format_optional_step(result.first_sample_collision_step),
                    format_optional_step(result.first_best_collision_step),
                    f"{result.max_sampled_collision_fraction:.3f}",
                )
            )
        )


def format_optional_step(step: int | None) -> str:
    return "none" if step is None else str(step)


def main() -> None:
    args = parse_args()
    results = [run_case(args, nsdf=False)]
    if args.include_nsdf:
        try:
            results.append(run_case(args, nsdf=True))
        except PretrainedSDFUnavailable as exc:
            print(f"skipping nsdf: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
