from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib
from matplotlib.animation import FFMpegWriter
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig, MPPIController
from robots import ROBOT_REGISTRY, create_robot
from sdf import PretrainedSDFUnavailable, default_obstacle_field, load_pretrained_sdf_for_robot


ROBOT_SHORTHANDS = {
    "single_integrator": "si",
    "unicycle": "uni",
    "dynamic_unicycle": "du",
    "planar_quadrotor": "quad2d",
    "mobile_arm": "mobile_arm",
}

DEMO_DEFAULTS = {
    "single_integrator": {"steps": 500, "horizon": 20, "samples": 80, "plot_samples": 80},
    "unicycle": {"steps": 500, "horizon": 36, "samples": 80, "plot_samples": 80},
    "dynamic_unicycle": {"steps": 500, "horizon": 20, "samples": 80, "plot_samples": 80},
    "planar_quadrotor": {"steps": 500, "horizon": 36, "samples": 80, "plot_samples": 80},
    "mobile_arm": {"steps": 500, "horizon": 28, "samples": 96, "plot_samples": 96},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a BR-MPPI obstacle-avoidance demo.")
    parser.add_argument("--algo", "--method", choices=ALGORITHMS, default="brmppi")
    parser.add_argument("--robot", "--dynamics", choices=tuple(sorted(ROBOT_REGISTRY)), default="unicycle")
    parser.add_argument("--nsdf", action="store_true", help="Use the pretrained neural signed distance model for h.")
    parser.add_argument(
        "--nsdf-variant",
        choices=("legacy", "retrained"),
        default="retrained",
        help="Checkpoint family selected by --nsdf (default: retrained).",
    )
    parser.add_argument("--mobile-nsdf-points", type=int, default=64)
    parser.add_argument("--mobile-nsdf-narrow-points", type=int, default=4)
    parser.add_argument(
        "--mobile-nsdf-analytic-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--steps", type=int, default=None, help="Simulation steps. Default depends on the robot.")
    parser.add_argument("--horizon", type=int, default=None, help="MPPI rollout horizon. Default depends on the robot.")
    parser.add_argument("--samples", type=int, default=None, help="Number of MPPI samples. Default depends on the robot.")
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=None,
        help="Number of sampled MPPI trajectories to draw. Default depends on the robot.",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true", help="Run without opening a Matplotlib window.")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image path.")
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save the rollout animation as an mp4 in output/animations.",
    )
    parser.add_argument("--animation-path", type=Path, default=None, help="Optional output mp4 path.")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second.")
    parser.add_argument("--animation-stride", type=int, default=2, help="Save every Nth rollout frame.")
    parser.add_argument("--collision-hold-seconds", type=float, default=3.0, help="Seconds to hold collision frame.")
    args = parser.parse_args()
    apply_demo_defaults(args)
    return args


def apply_demo_defaults(args: argparse.Namespace) -> None:
    # Analytic and neural runs intentionally use the same controller settings so
    # selecting --nsdf changes only the h-function implementation.
    defaults = DEMO_DEFAULTS[args.robot]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def load_selected_sdf(
    robot_name: str,
    algo: str,
    *,
    nsdf: bool,
    variant: str = "legacy",
    mobile_points: int = 64,
    mobile_narrow_points: int = 4,
    mobile_analytic_guard: bool = True,
):
    if not nsdf:
        return None
    if algo != "brmppi":
        raise NotImplementedError(
            f"--nsdf is only implemented for brmppi, not {algo}. "
            "Run this method without --nsdf or select --algo brmppi."
        )
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


def main() -> None:
    args = parse_args()
    if args.headless or args.save_animation:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    robot = create_robot(args.robot)
    field = default_obstacle_field(robot.name)
    sdf_model = load_selected_sdf(
        robot.name,
        args.algo,
        nsdf=args.nsdf,
        variant=args.nsdf_variant,
        mobile_points=args.mobile_nsdf_points,
        mobile_narrow_points=args.mobile_nsdf_narrow_points,
        mobile_analytic_guard=args.mobile_nsdf_analytic_guard,
    )
    if sdf_model is not None:
        print(f"pretrained_sdf={sdf_model.description}")
    config = MPPIConfig(horizon=args.horizon, samples=args.samples, dt=args.dt, plot_samples=args.plot_samples)
    controller = MPPIController(robot, field, algo=args.algo, sdf_model=sdf_model, config=config, seed=args.seed)

    state = robot.default_state.copy()
    goal = robot.default_goal.copy()
    trajectory = [state.copy()]
    sampled_rollouts = []
    best_rollouts = []
    min_exact_clearance = exact_clearance(field, robot, state)
    min_sampled_rollout_clearance = float("inf")
    min_best_rollout_clearance = float("inf")
    max_sampled_collision_fraction = 0.0
    first_sample_collision_step: int | None = None
    first_best_collision_step: int | None = None
    reached = False
    collision_index = 0 if min_exact_clearance < 0.0 else None

    for step_idx in range(args.steps):
        action, diagnostics = controller.command(state, goal)
        sampled_rollouts.append(diagnostics["sampled_trajectories"])
        best_rollouts.append(diagnostics["best_trajectory"])
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
        trajectory.append(state.copy())
        current_clearance = exact_clearance(field, robot, state)
        min_exact_clearance = min(min_exact_clearance, current_clearance)
        if current_clearance < 0.0:
            collision_index = len(trajectory) - 1
            break
        if float(jnp.linalg.norm(robot.position(state) - goal)) <= robot.goal_tolerance:
            reached = True
            break

    trajectory_arr = jnp.vstack(trajectory)
    final_error = float(jnp.linalg.norm(robot.position(state) - goal))
    collision = collision_index is not None
    output_path = args.save or default_plot_path(args)
    bounds = axis_bounds(field, robot, trajectory_arr, goal)
    should_show_animation = not args.headless and not args.save_animation
    plot_demo(
        output_path,
        field,
        robot,
        trajectory_arr,
        sampled_rollouts,
        best_rollouts,
        goal,
        state,
        args,
        bounds,
        collision_index,
        show_plot=not args.headless and not should_show_animation,
    )
    animation_path = None
    if args.save_animation:
        animation_path = args.animation_path or default_animation_path(args)
        save_animation(animation_path, field, robot, trajectory_arr, sampled_rollouts, best_rollouts, goal, args, bounds, collision_index)
    elif should_show_animation:
        show_animation(field, robot, trajectory_arr, sampled_rollouts, best_rollouts, goal, args, bounds, collision_index)

    print(
        f"algo={args.algo} robot={args.robot} nsdf={args.nsdf} "
        f"nsdf_variant={args.nsdf_variant if args.nsdf else 'none'} "
        f"barrier_source={controller.barrier_source}"
    )
    print(f"reached={reached} collision={collision} steps={len(trajectory_arr) - 1}")
    print(f"final_error={final_error:.3f} min_exact_clearance={min_exact_clearance:.3f}")
    print(
        "sampled_min_clearance="
        f"{min_sampled_rollout_clearance:.3f} best_rollout_min_clearance={min_best_rollout_clearance:.3f}"
    )
    print(
        "first_sample_collision_step="
        f"{format_optional_step(first_sample_collision_step)} "
        f"first_best_collision_step={format_optional_step(first_best_collision_step)} "
        f"max_sampled_collision_fraction={max_sampled_collision_fraction:.3f}"
    )
    print(f"plot={output_path}")
    if animation_path is not None:
        print(f"animation={animation_path}")


def exact_clearance(field, robot, state: jnp.ndarray) -> float:
    return float(jnp.min(field.signed_distance(robot.body_points(state))) - robot.body_point_radius)


def format_optional_step(step: int | None) -> str:
    return "none" if step is None else str(step)


def collision_point(field, robot, state: jnp.ndarray) -> jnp.ndarray:
    points = robot.body_points(state)
    distances = field.signed_distance(points)
    return points[int(jnp.argmin(distances))]


def plot_demo(
    output_path: Path,
    field,
    robot,
    trajectory: np.ndarray,
    sampled_rollouts: list[np.ndarray],
    best_rollouts: list[np.ndarray],
    goal: np.ndarray,
    final_state: np.ndarray,
    args,
    bounds: tuple[float, float, float, float],
    collision_index: int | None,
    show_plot: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_axisbelow(True)
    field.draw(ax)
    sampled, best = rollouts_for_frame(sampled_rollouts, best_rollouts, len(trajectory) - 1)
    draw_sampled_rollouts(ax, robot, sampled, best)
    positions = np.array([robot.position(s) for s in trajectory])
    ax.plot(positions[:, 0], positions[:, 1], color="tab:blue", linewidth=2.2, label="trajectory")
    ax.scatter([positions[0, 0]], [positions[0, 1]], color="tab:green", s=90, label="start", zorder=4)
    ax.scatter([goal[0]], [goal[1]], marker="*", color="tab:red", s=180, label="goal", zorder=5)
    robot.draw(ax, final_state, color="tab:orange", edgecolor="black")
    if collision_index is not None:
        draw_collision_marker(ax, field, robot, trajectory[collision_index], bounds)
    ax.set_title(f"{args.algo} / {args.robot} / {'pretrained NSDF' if args.nsdf else 'analytic SDF'}")
    set_axes(ax, bounds)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)


def default_plot_path(args) -> Path:
    suffix = f"_nsdf_{args.nsdf_variant}" if args.nsdf else ""
    return Path("output") / f"{robot_shorthand(args.robot)}_{args.algo}{suffix}.png"


def default_animation_path(args) -> Path:
    suffix = f"_nsdf_{args.nsdf_variant}" if args.nsdf else ""
    return Path("output") / "animations" / f"{robot_shorthand(args.robot)}_{args.algo}{suffix}.mp4"


def robot_shorthand(robot_name: str) -> str:
    return ROBOT_SHORTHANDS.get(robot_name, robot_name)


def save_animation(
    output_path: Path,
    field,
    robot,
    trajectory: np.ndarray,
    sampled_rollouts: list[np.ndarray],
    best_rollouts: list[np.ndarray],
    goal: np.ndarray,
    args,
    bounds: tuple[float, float, float, float],
    collision_index: int | None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_axisbelow(True)
    ffmpeg_path = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if Path(ffmpeg_path).exists():
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    writer = FFMpegWriter(fps=args.fps, metadata={"title": "BR-MPPI rollout", "artist": "br-mppi"})
    stride = max(1, args.animation_stride)
    frame_indices = list(range(0, len(trajectory), stride))
    if frame_indices[-1] != len(trajectory) - 1:
        frame_indices.append(len(trajectory) - 1)
    hold_frames = int(max(0.0, args.collision_hold_seconds) * args.fps) if collision_index is not None else 0

    with writer.saving(fig, str(output_path), dpi=160):
        for frame_idx in frame_indices:
            draw_frame(ax, field, robot, trajectory, sampled_rollouts, best_rollouts, goal, frame_idx, args, bounds, collision_index)
            writer.grab_frame()
        for _ in range(hold_frames):
            draw_frame(ax, field, robot, trajectory, sampled_rollouts, best_rollouts, goal, collision_index, args, bounds, collision_index)
            writer.grab_frame()
    plt.close(fig)


def show_animation(
    field,
    robot,
    trajectory: np.ndarray,
    sampled_rollouts: list[np.ndarray],
    best_rollouts: list[np.ndarray],
    goal: np.ndarray,
    args,
    bounds: tuple[float, float, float, float],
    collision_index: int | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.set_axisbelow(True)
    stride = max(1, args.animation_stride)
    frame_indices = list(range(0, len(trajectory), stride))
    if frame_indices[-1] != len(trajectory) - 1:
        frame_indices.append(len(trajectory) - 1)
    if collision_index is not None:
        frame_indices.extend([collision_index] * int(max(0.0, args.collision_hold_seconds) * args.fps))

    def update(frame_idx: int) -> None:
        draw_frame(ax, field, robot, trajectory, sampled_rollouts, best_rollouts, goal, frame_idx, args, bounds, collision_index)

    animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / max(1, args.fps), repeat=False)
    plt.show()
    _ = animation
    plt.close(fig)


def draw_frame(
    ax,
    field,
    robot,
    trajectory: np.ndarray,
    sampled_rollouts: list[np.ndarray],
    best_rollouts: list[np.ndarray],
    goal: np.ndarray,
    frame_idx: int,
    args,
    bounds: tuple[float, float, float, float],
    collision_index: int | None,
) -> None:
    ax.clear()
    ax.set_axisbelow(True)
    field.draw(ax)
    sampled, best = rollouts_for_frame(sampled_rollouts, best_rollouts, frame_idx)
    draw_sampled_rollouts(ax, robot, sampled, best)
    positions = np.array([robot.position(s) for s in trajectory[: frame_idx + 1]])
    ax.plot(positions[:, 0], positions[:, 1], color="tab:blue", linewidth=2.2, label="trajectory")
    ax.scatter([positions[0, 0]], [positions[0, 1]], color="tab:green", s=90, label="start", zorder=4)
    ax.scatter([goal[0]], [goal[1]], marker="*", color="tab:red", s=180, label="goal", zorder=5)
    robot.draw(ax, trajectory[frame_idx], color="tab:orange", edgecolor="black")
    if collision_index is not None and frame_idx >= collision_index:
        draw_collision_marker(ax, field, robot, trajectory[collision_index], bounds)
    ax.set_title(f"{args.algo} / {args.robot} / {'pretrained NSDF' if args.nsdf else 'analytic SDF'}")
    set_axes(ax, bounds)
    ax.grid(True, alpha=0.22, zorder=0)
    ax.legend(loc="upper left")


def draw_sampled_rollouts(ax, robot, sampled_rollouts: np.ndarray, best_rollout: np.ndarray) -> None:
    if sampled_rollouts.size:
        for rollout in sampled_rollouts:
            positions = np.array([robot.position(state) for state in rollout])
            ax.plot(positions[:, 0], positions[:, 1], color="tab:green", linewidth=0.8, alpha=0.2)
    if best_rollout.size:
        positions = np.array([robot.position(state) for state in best_rollout])
        ax.plot(positions[:, 0], positions[:, 1], color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.65)


def rollouts_for_frame(
    sampled_rollouts: list[np.ndarray],
    best_rollouts: list[np.ndarray],
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not sampled_rollouts:
        return np.array([]), np.array([])
    rollout_idx = min(frame_idx, len(sampled_rollouts) - 1)
    return sampled_rollouts[rollout_idx], best_rollouts[rollout_idx]


def draw_collision_marker(ax, field, robot, state: np.ndarray, bounds: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = bounds
    point = collision_point(field, robot, state)
    offset = 0.045 * min(x_max - x_min, y_max - y_min)
    ax.text(
        point[0] + offset,
        point[1] + offset,
        "!",
        color="red",
        fontsize=30,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=100,
    )


def axis_bounds(field, robot, trajectory: np.ndarray, goal: np.ndarray) -> tuple[float, float, float, float]:
    positions = np.array([robot.position(s) for s in trajectory])
    obstacle_centers = np.vstack([obs.center for obs in field.obstacles])
    radii = np.array([obs.radius for obs in field.obstacles])
    x_min = min(float(np.min(positions[:, 0])), float(goal[0]), float(np.min(obstacle_centers[:, 0] - radii)))
    x_max = max(float(np.max(positions[:, 0])), float(goal[0]), float(np.max(obstacle_centers[:, 0] + radii)))
    y_min = min(float(np.min(positions[:, 1])), float(goal[1]), float(np.min(obstacle_centers[:, 1] - radii)))
    y_max = max(float(np.max(positions[:, 1])), float(goal[1]), float(np.max(obstacle_centers[:, 1] + radii)))
    padding = 1.8 if robot.name == "mobile_arm" else 1.0
    return x_min - padding, x_max + padding, y_min - padding, y_max + padding


def set_axes(ax, bounds: tuple[float, float, float, float]) -> None:
    x_min, x_max, y_min, y_max = bounds
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


if __name__ == "__main__":
    main()
