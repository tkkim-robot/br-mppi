from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig, MPPIController, load_tuned_config
from robots import ROBOT_REGISTRY, create_robot
from sdf.hero_scenarios import load_hero_scenarios, get_hero_scenario
from examples.random_benchmark import timing_stats_ms, is_deadlocked, exact_clearance, TrialResult
from examples.basic_demo import axis_bounds, draw_frame, rollouts_for_frame, draw_sampled_rollouts, set_axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hero scenario evaluation and visualization.")
    parser.add_argument("--scenario", type=str, default="narrow_passage", help="Scenario name from hero_scenarios.yaml.")
    parser.add_argument("--robot-type", type=str, default=None, help="Override the robot dynamics (e.g., dynamic_unicycle, planar_quadrotor, mobile_arm).")
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS, help="List of algorithms to evaluate.")
    parser.add_argument("--nsdf", action="store_true", help="Use pretrained NSDF for brmppi.")
    parser.add_argument("--steps", type=int, default=600, help="Max simulation steps.")
    parser.add_argument("--seed", type=int, default=13, help="Controller seed.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Generate comparison video.")
    parser.add_argument("--save-figure", action="store_true", help="Generate summary comparison figure.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/hero"), help="Output directory.")
    return parser.parse_args()


def run_hero_trial(
    scenario_name: str,
    algo: str,
    nsdf: bool,
    max_steps: int,
    seed: int,
    robot_type: str | None = None,
) -> tuple[TrialResult, dict]:
    scenario = get_hero_scenario(scenario_name)
    
    if robot_type is not None and robot_type != scenario.robot_name:
        robot = create_robot(robot_type)
        new_start = jnp.zeros(robot.state_dim)
        if scenario.start_state is not None:
            copy_len = min(len(scenario.start_state), robot.state_dim)
            new_start = new_start.at[:copy_len].set(scenario.start_state[:copy_len])
        scenario = replace(scenario, robot_name=robot_type, start_state=new_start)
    else:
        robot = create_robot(scenario.robot_name)
    
    # Load tuned config
    try:
        config = load_tuned_config(scenario.robot_name, algo)
    except KeyError:
        print(f"Warning: No tuned config for {scenario.robot_name}/{algo}, using defaults.")
        config = MPPIConfig()

    # NSDF support
    sdf_model = None
    if nsdf and algo == "brmppi":
        from sdf import load_pretrained_sdf_for_robot
        sdf_model = load_pretrained_sdf_for_robot(scenario.robot_name, repo_root=REPO_ROOT)

    controller = MPPIController(
        robot,
        scenario.obstacle_field,
        algo=algo,
        sdf_model=sdf_model,
        config=config,
        seed=seed,
    )

    state = scenario.start_state if scenario.start_state is not None else robot.default_state.copy()
    goal = scenario.goal if scenario.goal is not None else robot.default_goal.copy()
    
    trajectory = [state.copy()]
    sampled_rollouts = []
    best_rollouts = []
    command_times = []
    
    min_exact_clearance = exact_clearance(scenario.obstacle_field, robot, state)
    min_sampled_rollout_clearance = float("inf")
    min_best_rollout_clearance = float("inf")
    max_sampled_collision_fraction = 0.0
    first_sample_collision_step: int | None = None
    first_best_collision_step: int | None = None
    reached = False
    collision = min_exact_clearance < 0.0
    deadlock = False
    steps_run = 0
    
    position_history = [np.asarray(robot.position(state), dtype=float)]
    goal_distance_history = [float(jnp.linalg.norm(robot.position(state) - goal))]
    
    wall_start = time.perf_counter()
    
    for step_idx in range(max_steps):
        steps_run = step_idx + 1
        cmd_start = time.perf_counter()
        action, diagnostics = controller.command(state, goal)
        jax.block_until_ready(action)
        command_times.append(time.perf_counter() - cmd_start)
        
        sampled_rollouts.append(diagnostics["sampled_trajectories"])
        best_rollouts.append(diagnostics["best_trajectory"])
        
        sampled_min = float(diagnostics["sampled_min_clearance"])
        best_min = float(diagnostics["best_min_clearance"])
        min_sampled_rollout_clearance = min(min_sampled_rollout_clearance, sampled_min)
        min_best_rollout_clearance = min(min_best_rollout_clearance, best_min)
        max_sampled_collision_fraction = max(max_sampled_collision_fraction, float(diagnostics["sampled_collision_fraction"]))
        
        if first_sample_collision_step is None and int(diagnostics["sampled_collision_count"]) > 0:
            first_sample_collision_step = step_idx
        if first_best_collision_step is None and bool(diagnostics["best_collision"]):
            first_best_collision_step = step_idx

        state = robot.step(state, action, config.dt)
        trajectory.append(state.copy())
        
        pos = np.asarray(robot.position(state), dtype=float)
        dist = float(jnp.linalg.norm(pos - goal))
        position_history.append(pos)
        goal_distance_history.append(dist)
        
        curr_clearance = exact_clearance(scenario.obstacle_field, robot, state)
        min_exact_clearance = min(min_exact_clearance, curr_clearance)
        
        if curr_clearance < 0.0:
            collision = True
            break
        if dist <= robot.goal_tolerance:
            reached = True
            break
        if is_deadlocked(position_history, goal_distance_history, deadlock_window=80, position_tolerance=0.03, progress_tolerance=0.02):
            deadlock = True
            break

    final_error = float(jnp.linalg.norm(robot.position(state) - goal))
    mean_ms, p95_ms = timing_stats_ms(command_times)
    
    result = TrialResult(
        trial=0, field_seed=0, robot=robot.name, algo=algo, obstacle_count=len(scenario.obstacle_field.obstacles),
        reached=reached, collision=collision, timeout=not reached and not collision, steps=steps_run,
        final_error=final_error, min_exact_clearance=float(min_exact_clearance),
        sampled_min_clearance=float(min_sampled_rollout_clearance),
        best_rollout_min_clearance=float(min_best_rollout_clearance),
        first_sample_collision_step=first_sample_collision_step,
        first_best_collision_step=first_best_collision_step,
        max_sampled_collision_fraction=float(max_sampled_collision_fraction),
        mean_command_ms=mean_ms, p95_command_ms=p95_ms,
        wall_seconds=time.perf_counter() - wall_start,
        deadlock=deadlock, barrier_source=controller.barrier_source
    )
    
    data = {
        "trajectory": np.array(trajectory),
        "sampled_rollouts": sampled_rollouts,
        "best_rollouts": best_rollouts,
        "goal": goal,
        "scenario": scenario
    }
    
    return result, data


def plot_hero_comparison(
    output_path: Path,
    results: list[TrialResult],
    all_data: list[dict],
    algos: list[str],
):
    num_algos = len(algos)
    
    # Create asymmetrical layout
    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(3, 5, wspace=0.3, hspace=0.4)
    
    axes = []
    
    # 1. Main BR-MPPI plot (index 0)
    main_ax = fig.add_subplot(grid[:, :3])
    axes.append(main_ax)
    
    # 2. Smaller plots
    for i in range(1, num_algos):
        row = (i - 1) // 2
        col = 3 + ((i - 1) % 2)
        small_ax = fig.add_subplot(grid[row, col])
        axes.append(small_ax)
        
    for i, (algo, res, data) in enumerate(zip(algos, results, all_data)):
        ax = axes[i]
        scenario = data["scenario"]
        robot = create_robot(scenario.robot_name)
        field = scenario.obstacle_field
        trajectory = data["trajectory"]
        goal = data["goal"]
        
        field.draw(ax)
        
        # Add rollouts for the final frame to the static plot too!
        sampled_rollouts = data["sampled_rollouts"]
        best_rollouts = data["best_rollouts"]
        sampled, best = rollouts_for_frame(sampled_rollouts, best_rollouts, len(trajectory) - 1)
        draw_sampled_rollouts(ax, robot, sampled, best)
        
        positions = np.array([robot.position(s) for s in trajectory])
        ax.plot(positions[:, 0], positions[:, 1], color="crimson", linewidth=3.5, alpha=0.8, zorder=7)
        ax.scatter([positions[0, 0]], [positions[0, 1]], color="mediumseagreen", s=60, zorder=9)
        ax.scatter([goal[0]], [goal[1]], marker="*", color="gold", edgecolor="darkgoldenrod", s=200, zorder=9)
        
        robot.draw(ax, trajectory[-1], color="dodgerblue", edgecolor="midnightblue", zorder=10)
        
        status = "Success" if res.reached else ("Collision" if res.collision else "Timeout")
        
        if i == 0:
            ax.set_title(f"{algo.upper()}\n{status} ({res.steps} steps)", fontsize=28, fontweight='bold', pad=15)
        else:
            ax.set_title(f"{algo.upper()}\n{status} ({res.steps} steps)", fontsize=14, fontweight='bold', pad=8)
        
        bounds = axis_bounds(field, robot, trajectory, goal)
        set_axes(ax, bounds)
        ax.grid(True, alpha=0.1)

    # Hide unused axes
    for i in range(num_algos, len(axes)):
        axes[i].axis("off")
        
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_hero_video(
    output_path: Path,
    results: list[TrialResult],
    all_data: list[dict],
    algos: list[str],
):
    from matplotlib.animation import FFMpegWriter
    import shutil
    
    num_algos = len(algos)
    
    # Create an asymmetrical grid layout to match the screenshot
    # Left side: 1 huge plot spanning 3 rows and 3 columns
    # Right side: 6 smaller plots in a 3x2 grid
    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(3, 5, wspace=0.3, hspace=0.4)
    
    axes = []
    
    # 1. Main BR-MPPI plot (Index 0 in algos list usually)
    main_ax = fig.add_subplot(grid[:, :3])
    axes.append(main_ax)
    
    # 2. Smaller plots for the other algorithms
    # There are max 6 other algorithms (indices 1 through 6)
    for i in range(1, num_algos):
        # Calculate grid position in the remaining 3x2 area
        row = (i - 1) // 2
        col = 3 + ((i - 1) % 2)
        small_ax = fig.add_subplot(grid[row, col])
        axes.append(small_ax)
        
    max_len = max(len(data["trajectory"]) for data in all_data)
    
    ffmpeg_path = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if Path(ffmpeg_path).exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        
    writer = FFMpegWriter(fps=12, metadata={"title": "Hero Scenario Comparison"})
    
    with writer.saving(fig, str(output_path), dpi=120):
        for frame_idx in range(0, max_len, 2):
            for i, (algo, res, data) in enumerate(zip(algos, results, all_data)):
                ax = axes[i]
                ax.clear()
                
                scenario = data["scenario"]
                robot = create_robot(scenario.robot_name)
                field = scenario.obstacle_field
                trajectory = data["trajectory"]
                goal = data["goal"]
                sampled_rollouts = data["sampled_rollouts"]
                best_rollouts = data["best_rollouts"]
                
                curr_idx = min(frame_idx, len(trajectory) - 1)
                
                field.draw(ax)
                sampled, best = rollouts_for_frame(sampled_rollouts, best_rollouts, curr_idx)
                draw_sampled_rollouts(ax, robot, sampled, best)
                
                positions = np.array([robot.position(s) for s in trajectory[:curr_idx + 1]])
                ax.plot(positions[:, 0], positions[:, 1], color="crimson", linewidth=3.5, alpha=0.8, zorder=7)
                ax.scatter([positions[0, 0]], [positions[0, 1]], color="mediumseagreen", s=60, zorder=9)
                ax.scatter([goal[0]], [goal[1]], marker="*", color="gold", edgecolor="darkgoldenrod", s=200, zorder=9)
                
                robot.draw(ax, trajectory[curr_idx], color="dodgerblue", edgecolor="midnightblue", zorder=10)
                
                status = "Success" if (res.reached and curr_idx == len(trajectory)-1) else ("Collision" if (res.collision and curr_idx == len(trajectory)-1) else "")
                
                if i == 0:
                    ax.set_title(f"{algo.upper()} {status}", fontsize=28, fontweight='bold', pad=15)
                else:
                    ax.set_title(f"{algo.upper()} {status}", fontsize=14, fontweight='bold', pad=8)
                
                bounds = axis_bounds(field, robot, trajectory, goal)
                set_axes(ax, bounds)
                ax.grid(True, alpha=0.1)

            for i in range(num_algos, len(axes)):
                axes[i].axis("off")
            
            writer.grab_frame()
    plt.close(fig)


def main():
    args = parse_args()
    if args.headless or args.save_video or args.save_figure:
        matplotlib.use("Agg")
    
    scenarios = load_hero_scenarios()
    if args.scenario not in scenarios:
        print(f"Error: Unknown scenario {args.scenario}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        return

    print(f"Running hero scenario: {args.scenario}")
    print(f"Description: {scenarios[args.scenario].description}")
    
    all_results = []
    all_data = []
    
    for algo in args.algos:
        print(f"Evaluating {algo}...")
        res, data = run_hero_trial(args.scenario, algo, args.nsdf, args.steps, args.seed, robot_type=args.robot_type)
        all_results.append(res)
        all_data.append(data)
        print(f"  Result: {'Reached' if res.reached else 'Collision' if res.collision else 'Timeout'} in {res.steps} steps. Min clearance: {res.min_exact_clearance:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine the robot type used (either from args or scenario default)
    actual_robot = args.robot_type if args.robot_type else all_data[0]["scenario"].robot_name
    
    # Generate a unique timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = args.output_dir / f"{args.scenario}_{actual_robot}_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"Results saved to {json_path}")

    # Save Figure
    if args.save_figure:
        fig_path = args.output_dir / f"{args.scenario}_{actual_robot}_comparison_{timestamp}.png"
        plot_hero_comparison(fig_path, all_results, all_data, args.algos)
        print(f"Figure saved to {fig_path}")
        
    # Save Video
    if args.save_video:
        video_path = args.output_dir / f"{args.scenario}_{actual_robot}_comparison_{timestamp}.mp4"
        save_hero_video(video_path, all_results, all_data, args.algos)
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
