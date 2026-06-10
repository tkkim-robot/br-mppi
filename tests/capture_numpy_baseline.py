from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig, MPPIController
from robots import ROBOT_REGISTRY, create_robot
from sdf import default_obstacle_field


@dataclass(frozen=True)
class StepTrace:
    step: int
    state: list[float]
    action: list[float]
    next_state: list[float]
    cost: float
    mean_cost: float
    best_min_clearance: float
    sampled_min_clearance: float
    sampled_collision_fraction: float


@dataclass(frozen=True)
class BaselineTrace:
    robot: str
    algo: str
    seed: int
    steps: int
    horizon: int
    samples: int
    dt: float
    noise_scale: float
    alpha_noise_scale: float
    wall_seconds_per_command: list[float]
    traces: list[StepTrace]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture deterministic NumPy MPPI traces for JAX parity tests.")
    parser.add_argument("--robot", choices=tuple(sorted(ROBOT_REGISTRY)), default="unicycle")
    parser.add_argument("--algo", choices=ALGORITHMS, default="brmppi")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--alpha-noise-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = create_robot(args.robot)
    field = default_obstacle_field(robot.name)
    config_kwargs = {"horizon": args.horizon, "samples": args.samples, "dt": args.dt, "plot_samples": 0}
    if args.noise_scale is not None:
        config_kwargs["noise_scale"] = args.noise_scale
    if args.alpha_noise_scale is not None:
        config_kwargs["alpha_noise_scale"] = args.alpha_noise_scale
    config = MPPIConfig(**config_kwargs)
    controller = MPPIController(robot, field, algo=args.algo, config=config, seed=args.seed)
    state = robot.default_state.copy()
    goal = robot.default_goal.copy()
    traces: list[StepTrace] = []
    wall_times: list[float] = []

    for step_idx in range(args.steps):
        start = time.perf_counter()
        action, diagnostics = controller.command(state, goal)
        wall_times.append(time.perf_counter() - start)
        next_state = robot.step(state, action, args.dt)
        traces.append(
            StepTrace(
                step=step_idx,
                state=np.asarray(state, dtype=float).tolist(),
                action=np.asarray(action, dtype=float).tolist(),
                next_state=np.asarray(next_state, dtype=float).tolist(),
                cost=float(diagnostics["best_cost"]),
                mean_cost=float(diagnostics["mean_cost"]),
                best_min_clearance=float(diagnostics["best_min_clearance"]),
                sampled_min_clearance=float(diagnostics["sampled_min_clearance"]),
                sampled_collision_fraction=float(diagnostics["sampled_collision_fraction"]),
            )
        )
        state = next_state

    baseline = BaselineTrace(
        robot=robot.name,
        algo=args.algo,
        seed=args.seed,
        steps=args.steps,
        horizon=args.horizon,
        samples=args.samples,
        dt=args.dt,
        noise_scale=config.noise_scale,
        alpha_noise_scale=config.alpha_noise_scale,
        wall_seconds_per_command=wall_times,
        traces=traces,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(baseline), indent=2) + "\n")


if __name__ == "__main__":
    main()
