from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import MPPIConfig, MPPIController
from robots import create_robot
from sdf import default_obstacle_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare JAX command time against captured NumPy baseline JSON.")
    parser.add_argument("--baseline", type=Path, default=Path("tests/baselines/numpy_mobile_brmppi_timing.json"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    robot = create_robot(baseline["robot"])
    field = default_obstacle_field(robot.name)
    config = MPPIConfig(
        horizon=baseline["horizon"],
        samples=baseline["samples"],
        dt=baseline["dt"],
        plot_samples=0,
    )
    controller = MPPIController(robot, field, algo=baseline["algo"], config=config, seed=baseline["seed"])
    steps = args.steps or baseline["steps"]

    state = robot.default_state.copy()
    for _ in range(args.warmup):
        action, _diagnostics = controller.command(state, robot.default_goal)
        state = robot.step(state, action, baseline["dt"])
    jax.block_until_ready(state)

    times: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        action, _diagnostics = controller.command(state, robot.default_goal)
        jax.block_until_ready(action)
        times.append(time.perf_counter() - start)
        state = robot.step(state, action, baseline["dt"])

    numpy_times = baseline["wall_seconds_per_command"]
    report = {
        "robot": baseline["robot"],
        "algo": baseline["algo"],
        "horizon": baseline["horizon"],
        "samples": baseline["samples"],
        "dt": baseline["dt"],
        "numpy_mean_seconds": sum(numpy_times) / len(numpy_times),
        "jax_mean_seconds_after_warmup": sum(times) / len(times),
        "jax_seconds": times,
        "speedup_after_warmup": (sum(numpy_times) / len(numpy_times)) / (sum(times) / len(times)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
