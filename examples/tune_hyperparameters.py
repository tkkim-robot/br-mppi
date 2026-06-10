from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    from optuna.samplers import GPSampler
except ImportError:  # pragma: no cover - depends on Optuna version.
    GPSampler = None

try:
    import wandb
except ImportError:  # pragma: no cover - dependency is declared in pyproject.
    wandb = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig
from random_benchmark import ROBOT_DEFAULTS, random_obstacle_field, run_trial, summarize_results
from robots import ROBOT_REGISTRY, create_robot


CONFIG_FIELD_NAMES = {field.name for field in fields(MPPIConfig)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune MPPI-family hyperparameters with Optuna and W&B.")
    parser.add_argument("--algo", choices=ALGORITHMS, default="brmppi")
    parser.add_argument("--robot", "--dynamics", choices=tuple(sorted(ROBOT_REGISTRY)), default="unicycle")
    parser.add_argument("--optuna-trials", type=int, default=100, help="Number of Optuna hyperparameter trials.")
    parser.add_argument("--benchmark-trials", type=int, default=100, help="Random obstacle trials per Optuna trial.")
    parser.add_argument("--final-eval-trials", type=int, default=100, help="Benchmark trials for final best-config table.")
    parser.add_argument("--final-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=13, help="Base seed for benchmark obstacle fields.")
    parser.add_argument("--controller-seed", type=int, default=7, help="Base seed for MPPI sampling.")
    parser.add_argument("--horizon", type=int, default=None, help="Fixed horizon override before tuning.")
    parser.add_argument("--samples", type=int, default=None, help="Fixed sample-count override before tuning.")
    parser.add_argument("--max-steps", type=int, default=None, help="Generous per-trial simulation cap.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--min-obstacles", type=int, default=None)
    parser.add_argument("--max-obstacles", type=int, default=None)
    parser.add_argument("--workspace-margin", type=float, default=2.5)
    parser.add_argument("--start-clearance", type=float, default=0.35)
    parser.add_argument("--goal-clearance", type=float, default=0.9)
    parser.add_argument("--fixed-config", type=Path, default=None, help="Best BR config JSON for baseline fine tuning.")
    parser.add_argument(
        "--tune-shared",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tune shared MPPI parameters. Auto: true for brmppi or when no fixed config is supplied.",
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URI, e.g. sqlite:///output/tuning/foo.db")
    parser.add_argument("--sampler", choices=("gp", "tpe"), default="gp")
    parser.add_argument("--n-startup-trials", type=int, default=15)
    parser.add_argument("--prune-interval", type=int, default=10)
    parser.add_argument("--timeout-hours", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", type=str, default="br-mppi-tuning")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--output-dir", type=Path, default=Path("output/tuning"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    defaults = ROBOT_DEFAULTS[args.robot]
    max_steps = args.max_steps or defaults["max_steps"]
    min_obstacles = args.min_obstacles or defaults["min_obstacles"]
    max_obstacles = args.max_obstacles or defaults["max_obstacles"]
    study_name = args.study_name or f"{args.robot}_{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}"
    storage = args.storage or f"sqlite:///{args.output_dir / (study_name + '.db')}"
    tune_shared = args.tune_shared if args.tune_shared is not None else (args.algo == "brmppi" or args.fixed_config is None)
    base_config = load_base_config(args, defaults)
    run = init_wandb(args, study_name, base_config, tune_shared)

    optimizer = HyperparameterOptimizer(
        args=args,
        base_config=base_config,
        max_steps=max_steps,
        min_obstacles=min_obstacles,
        max_obstacles=max_obstacles,
        tune_shared=tune_shared,
        wandb_run=run,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=create_sampler(args),
        pruner=MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=max(1, args.prune_interval),
            interval_steps=max(1, args.prune_interval),
        ),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        optimizer.objective,
        n_trials=args.optuna_trials,
        timeout=None if args.timeout_hours is None else int(args.timeout_hours * 3600),
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    best_config = optimizer.config_from_trial(study.best_trial)
    save_tuning_outputs(args, study, best_config, optimizer.completed_results)

    if args.final_eval:
        final_results = optimizer.evaluate_config(
            best_config,
            optuna_trial=None,
            benchmark_trials=args.final_eval_trials,
            report_pruning=False,
        )
        write_final_markdown(args, study, best_config, final_results)
        if run is not None:
            summary = summarize_results(final_results)
            wandb.log({f"final_{key}": value for key, value in summary[0].items() if isinstance(value, (int, float))})

    if run is not None:
        wandb.finish()


class HyperparameterOptimizer:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        base_config: MPPIConfig,
        max_steps: int,
        min_obstacles: int,
        max_obstacles: int,
        tune_shared: bool,
        wandb_run: Any | None,
    ) -> None:
        self.args = args
        self.base_config = base_config
        self.max_steps = max_steps
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.tune_shared = tune_shared
        self.wandb_run = wandb_run
        self.completed_results: list[dict[str, Any]] = []
        self.lock_horizon = args.horizon is not None
        self.lock_samples = args.samples is not None

    def objective(self, trial: optuna.Trial) -> float:
        config = self.config_from_trial(trial)
        results = self.evaluate_config(
            config,
            optuna_trial=trial,
            benchmark_trials=self.args.benchmark_trials,
            report_pruning=True,
        )
        metrics = objective_metrics(results, config.dt, self.max_steps)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("config", config_to_dict(config))
        trial.set_user_attr("results", [asdict(result) for result in results])
        trial_record = {
            "trial_number": trial.number,
            "params": trial.params,
            "config": config_to_dict(config),
            "metrics": metrics,
        }
        self.completed_results.append(trial_record)
        if self.wandb_run is not None:
            wandb.log(
                {
                    "trial_number": trial.number,
                    **metrics,
                    **{f"param/{key}": value for key, value in trial.params.items()},
                },
                step=trial.number,
            )
        print(
            f"trial={trial.number:03d} score={metrics['objective_score']:.5f} "
            f"success={metrics['success_rate']:.3f} collision={metrics['collision_rate']:.3f} "
            f"avg_success_time={metrics['avg_success_time']:.2f}s params={trial.params}",
            flush=True,
        )
        return metrics["objective_score"]

    def config_from_trial(self, trial: optuna.Trial) -> MPPIConfig:
        params: dict[str, Any] = {}
        if self.tune_shared:
            params.update(
                suggest_shared_params(
                    trial,
                    self.base_config,
                    lock_horizon=self.lock_horizon,
                    lock_samples=self.lock_samples,
                )
            )
        params.update(suggest_algo_params(trial, self.args.algo, self.base_config))
        return update_config(self.base_config, params)

    def evaluate_config(
        self,
        config: MPPIConfig,
        *,
        optuna_trial: optuna.Trial | None,
        benchmark_trials: int,
        report_pruning: bool,
    ) -> list:
        robot_for_fields = create_robot(self.args.robot)
        results = []
        for benchmark_idx in range(benchmark_trials):
            field_seed = self.args.seed + benchmark_idx
            field = random_obstacle_field(
                robot_for_fields,
                np.random.default_rng(field_seed),
                min_obstacles=self.min_obstacles,
                max_obstacles=self.max_obstacles,
                workspace_margin=self.args.workspace_margin,
                start_clearance=self.args.start_clearance,
                goal_clearance=self.args.goal_clearance,
            )
            result = run_trial(
                robot_name=self.args.robot,
                field=field,
                algo=self.args.algo,
                trial=benchmark_idx,
                field_seed=field_seed,
                controller_seed=self.args.controller_seed + benchmark_idx,
                horizon=config.horizon,
                samples=config.samples,
                max_steps=self.max_steps,
                dt=config.dt,
                warmup=self.args.warmup,
                config=config,
            )
            results.append(result)
            if report_pruning and optuna_trial is not None and (benchmark_idx + 1) % self.args.prune_interval == 0:
                partial_metrics = objective_metrics(results, config.dt, self.max_steps)
                optuna_trial.report(partial_metrics["success_rate"], step=benchmark_idx + 1)
                if optuna_trial.should_prune():
                    print(
                        f"trial={optuna_trial.number:03d} pruned after {benchmark_idx + 1} "
                        f"benchmark trials with success={partial_metrics['success_rate']:.3f}",
                        flush=True,
                    )
                    raise optuna.TrialPruned()
        return results


def suggest_shared_params(
    trial: optuna.Trial,
    base: MPPIConfig,
    *,
    lock_horizon: bool,
    lock_samples: bool,
) -> dict[str, Any]:
    params = {
        "horizon": (
            base.horizon
            if lock_horizon
            else trial.suggest_categorical("horizon", [16, 20, 24, 28, 32, 36, 44, 52])
        ),
        "samples": (
            base.samples
            if lock_samples
            else trial.suggest_categorical("samples", [100, 160, 240, 320, 480, 640, 800, 1000])
        ),
        "temperature": trial.suggest_float("temperature", 0.5, 8.0, step=0.25),
        "noise_scale": trial.suggest_float("noise_scale", 0.1, 1.2, step=0.05),
        "goal_weight": trial.suggest_float("goal_weight", 0.25, 4.0, step=0.25),
        "final_goal_weight": trial.suggest_float("final_goal_weight", 2.0, 24.0, step=1.0),
        "control_weight": trial.suggest_float("control_weight", 0.0, 0.2, step=0.01),
        "barrier_projection_margin": trial.suggest_float("barrier_projection_margin", 0.02, 0.30, step=0.02),
        "collision_weight": trial.suggest_categorical("collision_weight", [1000.0, 3000.0, 10000.0, 30000.0]),
        "dt": base.dt,
    }
    return params


def suggest_algo_params(trial: optuna.Trial, algo: str, base: MPPIConfig) -> dict[str, Any]:
    if algo == "brmppi":
        return {
            "barrier_alpha": trial.suggest_float("barrier_alpha", 0.0, 2.0, step=0.1),
            "alpha_rate_bound": trial.suggest_float("alpha_rate_bound", 2.0, 30.0, step=1.0),
            "alpha_state_bound": trial.suggest_float("alpha_state_bound", 0.5, 5.0, step=0.25),
            "alpha_noise_scale": trial.suggest_float("alpha_noise_scale", 0.1, 1.5, step=0.05),
            "alpha_projection_inverse_weight": trial.suggest_categorical(
                "alpha_projection_inverse_weight",
                [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00],
            ),
            "bound_penalty": trial.suggest_categorical("bound_penalty", [0.0, 0.01, 0.05, 0.10, 0.50, 1.00]),
            "barrier_buffer_distance": trial.suggest_float("barrier_buffer_distance", 0.05, 0.60, step=0.05),
            "barrier_alpha_cost_weight": trial.suggest_float("barrier_alpha_cost_weight", 0.0, 0.10, step=0.005),
            "br_clearance_margin": trial.suggest_float("br_clearance_margin", 0.10, 0.80, step=0.05),
            "br_clearance_weight": trial.suggest_categorical("br_clearance_weight", [25.0, 50.0, 100.0, 150.0, 250.0, 400.0, 800.0]),
            "br_collision_weight": trial.suggest_categorical(
                "br_collision_weight",
                [10000.0, 30000.0, 100000.0, 300000.0, 1000000.0],
            ),
        }
    if algo == "penalty_mppi":
        collision_weight = trial.suggest_categorical(
            "penalty_collision_weight",
            [1000.0, 3000.0, 10000.0, 30000.0, 100000.0],
        )
        return {
            "safety_weight": trial.suggest_categorical("safety_weight", [5.0, 10.0, 18.0, 30.0, 60.0, 120.0]),
            "collision_weight": collision_weight,
        }
    if algo == "mppi_cbf":
        return {
            "cbf_alpha": trial.suggest_float("cbf_alpha", 0.2, 4.0, step=0.1),
            "cbf_qp_iterations": trial.suggest_categorical("cbf_qp_iterations", [20, 35, 50, 75, 100]),
            "cbf_qp_rho": trial.suggest_categorical("cbf_qp_rho", [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
            "cbf_qp_regularization": trial.suggest_categorical("cbf_qp_regularization", [1e-9, 1e-8, 1e-7, 1e-6]),
        }
    if algo == "shield_mppi":
        return {
            "shield_cbf_penalty_weight": trial.suggest_categorical("shield_cbf_penalty_weight", [100.0, 300.0, 900.0, 3000.0, 10000.0]),
            "shield_alpha": trial.suggest_float("shield_alpha", 0.80, 1.20, step=0.02),
            "shield_repair_horizon": trial.suggest_categorical("shield_repair_horizon", [4, 6, 8, 10, 12]),
            "shield_repair_steps": trial.suggest_categorical("shield_repair_steps", [4, 8, 12, 16, 24]),
            "shield_repair_step_size": trial.suggest_float("shield_repair_step_size", 0.01, 0.15, step=0.01),
        }
    if algo == "sc_mppi":
        return {
            "sc_barrier_eps": trial.suggest_categorical("sc_barrier_eps", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
            "sc_barrier_gamma": trial.suggest_float("sc_barrier_gamma", 0.0, 0.8, step=0.05),
            "sc_feedback_clip": trial.suggest_float("sc_feedback_clip", 0.2, 2.0, step=0.1),
            "sc_feedback_iterations": trial.suggest_categorical("sc_feedback_iterations", [1, 2, 3]),
            "sc_feedback_regularization": trial.suggest_categorical("sc_feedback_regularization", [1e-5, 1e-4, 1e-3, 1e-2]),
            "sc_beta_cost_weight": trial.suggest_categorical("sc_beta_cost_weight", [1.0, 3.0, 8.0, 20.0, 50.0]),
            "sc_state_cost_weight": trial.suggest_categorical("sc_state_cost_weight", [0.0, 0.02, 0.05, 0.10, 0.20]),
            "sc_control_cost_weight": trial.suggest_categorical("sc_control_cost_weight", [0.02, 0.05, 0.08, 0.15, 0.30]),
        }
    if algo == "gs_mppi":
        return {
            "gs_softmin_rho": trial.suggest_categorical("gs_softmin_rho", [5.0, 10.0, 15.0, 25.0, 40.0, 60.0]),
            "gs_closed_form_gamma": trial.suggest_float("gs_closed_form_gamma", 0.2, 6.0, step=0.2),
            "gs_alpha": trial.suggest_float("gs_alpha", 0.2, 4.0, step=0.1),
            "gs_composite_margin": trial.suggest_float("gs_composite_margin", 0.0, 0.30, step=0.02),
        }
    return {}


def objective_metrics(results: list, dt: float, max_steps: int) -> dict[str, float]:
    total = max(len(results), 1)
    successes = sum(result.reached for result in results)
    collisions = sum(result.collision for result in results)
    timeouts = sum(result.timeout for result in results)
    success_times = [result.steps * dt for result in results if result.reached]
    max_time = max_steps * dt
    avg_success_time = sum(success_times) / len(success_times) if success_times else max_time
    travel_bonus = 0.001 * max(0.0, 1.0 - avg_success_time / max(max_time, 1e-9))
    success_rate = successes / total
    return {
        "objective_score": success_rate + travel_bonus,
        "success_rate": success_rate,
        "collision_rate": collisions / total,
        "timeout_rate": timeouts / total,
        "avg_success_time": avg_success_time,
        "mean_steps": sum(result.steps for result in results) / total,
        "mean_command_ms": sum(result.mean_command_ms for result in results) / total,
        "mean_final_error": sum(result.final_error for result in results) / total,
        "worst_min_clearance": min(result.min_exact_clearance for result in results),
    }


def load_base_config(args: argparse.Namespace, defaults: dict[str, int]) -> MPPIConfig:
    config = MPPIConfig(
        horizon=args.horizon or defaults["horizon"],
        samples=args.samples or defaults["samples"],
        dt=args.dt,
        plot_samples=0,
    )
    if args.fixed_config is not None:
        config = update_config(config, json.loads(args.fixed_config.read_text(encoding="utf-8")))
    return config


def update_config(config: MPPIConfig, params: dict[str, Any]) -> MPPIConfig:
    clean_params = {key: value for key, value in params.items() if key in CONFIG_FIELD_NAMES}
    return MPPIConfig(**{**config_to_dict(config), **clean_params, "plot_samples": 0})


def config_to_dict(config: MPPIConfig) -> dict[str, Any]:
    return asdict(config)


def create_sampler(args: argparse.Namespace):
    if args.sampler == "gp" and GPSampler is not None:
        return GPSampler(seed=args.seed)
    return TPESampler(seed=args.seed, multivariate=True)


def init_wandb(args: argparse.Namespace, study_name: str, base_config: MPPIConfig, tune_shared: bool):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise RuntimeError("wandb is not installed. Run `uv sync` first.")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=study_name,
        mode=args.wandb_mode,
        config={
            "algo": args.algo,
            "robot": args.robot,
            "optuna_trials": args.optuna_trials,
            "benchmark_trials": args.benchmark_trials,
            "seed": args.seed,
            "controller_seed": args.controller_seed,
            "tune_shared": tune_shared,
            "base_config": config_to_dict(base_config),
        },
    )


def save_tuning_outputs(args: argparse.Namespace, study: optuna.Study, best_config: MPPIConfig, records: list[dict[str, Any]]) -> None:
    best_config_path = args.output_dir / f"{study.study_name}_best_config.json"
    best_config_path.write_text(json.dumps(config_to_dict(best_config), indent=2), encoding="utf-8")
    report_path = args.output_dir / f"{study.study_name}_study.json"
    report = {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config_path": str(best_config_path),
        "trials": records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"best_config={best_config_path}")
    print(f"study_report={report_path}")


def write_final_markdown(args: argparse.Namespace, study: optuna.Study, best_config: MPPIConfig, results: list) -> None:
    summary = summarize_results(results)
    benchmark_json = args.output_dir / f"{study.study_name}_best_benchmark.json"
    benchmark_json.write_text(
        json.dumps(
            {
                "best_config": config_to_dict(best_config),
                "summary": summary,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown_path = args.output_dir / f"{study.study_name}_best_benchmark.md"
    lines = [
        f"# Best Benchmark: {args.robot} / {args.algo}",
        "",
        f"- Study: `{study.study_name}`",
        f"- Best trial: `{study.best_trial.number}`",
        f"- Best objective: `{study.best_value:.6f}`",
        f"- Benchmark trials: `{len(results)}`",
        f"- JSON: `{benchmark_json}`",
        "",
        "| dynamics | method | trials | reached | collisions | timeouts | success rate | collision rate | mean reach/trial steps | mean command ms | worst clearance |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                (
                    args.robot,
                    str(row["algo"]),
                    str(row["trials"]),
                    str(row["reached"]),
                    str(row["collisions"]),
                    str(row["timeouts"]),
                    f"{float(row['success_rate']):.3f}",
                    f"{float(row['collision_rate']):.3f}",
                    f"{float(row['mean_steps']):.1f}",
                    f"{float(row['mean_command_ms']):.2f}",
                    f"{float(row['worst_min_clearance']):+.3f}",
                )
            )
            + " |"
        )
    lines.extend(["", "## Best Config", "", "```json", json.dumps(config_to_dict(best_config), indent=2), "```"])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"final_benchmark={benchmark_json}")
    print(f"final_markdown={markdown_path}")


if __name__ == "__main__":
    main()
