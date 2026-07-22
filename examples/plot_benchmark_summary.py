from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


_DYNAMICS = (
    "single_integrator",
    "unicycle",
    "dynamic_unicycle",
    "planar_quadrotor",
    "mobile_arm",
)
_METHODS = (
    "brmppi",
    "penalty_mppi",
    "mppi_cbf",
    "shield_mppi",
    "sc_mppi",
    "gs_mppi",
)
_OPTIONAL_METHODS = ("mppi",)

_DYNAMICS_LABELS = {
    "single_integrator": "Single integrator",
    "unicycle": "Unicycle",
    "dynamic_unicycle": "Dynamic unicycle",
    "planar_quadrotor": "Planar quadrotor",
    "mobile_arm": "Mobile arm",
}
_METHOD_LABELS = {
    "brmppi": "BR-MPPI",
    "mppi": "MPPI",
    "penalty_mppi": "Penalty MPPI",
    "mppi_cbf": "MPPI-CBF",
    "shield_mppi": "Shield MPPI",
    "sc_mppi": "SC-MPPI",
    "gs_mppi": "GS-MPPI",
}

_OUTCOME_COLORS = {
    "success": "#3F7CFF",
    "collision": "#F04F70",
    "deadlock_timeout": "#8A63D2",
    "other_timeout": "#D9DEE7",
}
_TIME_COLORS = {
    "brmppi": "#263A8B",
    "penalty_mppi": "#2BA37F",
    "mppi_cbf": "#E38B29",
    "shield_mppi": "#C85082",
    "sc_mppi": "#7B68B8",
    "gs_mppi": "#6F7E8C",
    "mppi": "#A5A5A5",
}


@dataclass(frozen=True)
class BenchmarkRow:
    dynamics: str
    method: str
    trials: int
    reached: int
    collisions: int
    timeouts: int
    deadlocks: int
    success_rate: float
    collision_rate: float
    mean_command_ms: float

    @property
    def deadlock_timeouts(self) -> int:
        return min(self.deadlocks, self.timeouts)

    @property
    def other_timeouts(self) -> int:
        return max(self.timeouts - self.deadlock_timeouts, 0)

    @property
    def label(self) -> str:
        return f"{_DYNAMICS_LABELS[self.dynamics]} | {_METHOD_LABELS[self.method]}"


def _read_rows(tuning_dir: Path, methods: Iterable[str]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for dynamics in _DYNAMICS:
        for method in methods:
            path = tuning_dir / f"{dynamics}_{method}_best_benchmark.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing benchmark artifact: {path}")
            data = json.loads(path.read_text())
            summary = data["summary"][0]
            rows.append(
                BenchmarkRow(
                    dynamics=dynamics,
                    method=method,
                    trials=int(summary["trials"]),
                    reached=int(summary["reached"]),
                    collisions=int(summary["collisions"]),
                    timeouts=int(summary["timeouts"]),
                    deadlocks=int(summary.get("deadlocks", 0)),
                    success_rate=float(summary["success_rate"]),
                    collision_rate=float(summary["collision_rate"]),
                    mean_command_ms=float(summary["mean_command_ms"]),
                )
            )
    return rows


def _row_positions(rows: list[BenchmarkRow]) -> np.ndarray:
    positions: list[float] = []
    y = 0.0
    previous_dynamics = None
    for row in rows:
        if previous_dynamics is not None and row.dynamics != previous_dynamics:
            y += 0.75
        positions.append(y)
        y += 1.0
        previous_dynamics = row.dynamics
    return np.asarray(positions)


def _add_group_labels(ax: plt.Axes, rows: list[BenchmarkRow], y: np.ndarray) -> None:
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end].dynamics == rows[start].dynamics:
            end += 1
        center = float((y[start] + y[end - 1]) / 2.0)
        ax.text(
            -0.17,
            center,
            _DYNAMICS_LABELS[rows[start].dynamics],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        if end < len(rows):
            ax.axhline(y[end] - 0.52, color="#D8DDE8", linewidth=0.8)
        start = end


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#C8CFDC", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#B6BDCB")


def _annotate_outcome_segment(
    ax: plt.Axes,
    x_left: float,
    width: float,
    y: float,
    count: int,
    color: str,
    text_color: str,
) -> None:
    if count <= 0:
        return
    label = str(count)
    if width >= 5.0:
        ax.text(
            x_left + width / 2.0,
            y,
            label,
            ha="center",
            va="center",
            color=text_color,
            fontsize=8,
            fontweight="bold",
        )
    else:
        ax.text(
            min(x_left + width + 0.7, 104.5),
            y,
            label,
            ha="left",
            va="center",
            color=color,
            fontsize=7,
            fontweight="bold",
            clip_on=False,
        )


def _plot_outcomes(rows: list[BenchmarkRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 11.2))
    y = _row_positions(rows)
    left = np.zeros(len(rows))
    height = 0.68

    segments = (
        ("success", "Success", [row.reached for row in rows], "white"),
        ("collision", "Collision", [row.collisions for row in rows], "white"),
        ("deadlock_timeout", "Deadlock timeout", [row.deadlock_timeouts for row in rows], "white"),
        ("other_timeout", "Other timeout", [row.other_timeouts for row in rows], "#2C3440"),
    )

    for key, label, counts, text_color in segments:
        widths = np.asarray([100.0 * count / row.trials for count, row in zip(counts, rows)])
        color = _OUTCOME_COLORS[key]
        ax.barh(y, widths, height, left=left, color=color, label=label)
        for idx, (width, count) in enumerate(zip(widths, counts)):
            _annotate_outcome_segment(ax, left[idx], width, y[idx], count, color, text_color)
        left += widths

    ax.set_xlim(0, 106)
    ax.set_ylim(y[-1] + 0.8, -0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_METHOD_LABELS[row.method] for row in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Outcome rate (%)")
    ax.set_title("Tuned Benchmark Outcomes", fontsize=15, fontweight="bold", pad=16)
    ax.text(
        0.0,
        1.01,
        "Numbers inside each bar are trial counts. Deadlocks are a subset of timeouts, so timeout failures are split.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    _add_group_labels(ax, rows, y)
    _style_axis(ax)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        frameon=False,
        handles=[Patch(facecolor=_OUTCOME_COLORS[key], label=label) for key, label, _, _ in segments],
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_command_time(rows: list[BenchmarkRow], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 11.2))
    y = _row_positions(rows)
    values = np.asarray([row.mean_command_ms for row in rows])
    colors = [_TIME_COLORS[row.method] for row in rows]
    ax.barh(y, values, 0.68, color=colors)

    max_value = float(np.max(values))
    for yi, value in zip(y, values):
        ax.text(
            value + max_value * 0.012,
            yi,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=8,
            color="#2C3440",
        )

    ax.set_xlim(0, max_value * 1.18)
    ax.set_ylim(y[-1] + 0.8, -0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_METHOD_LABELS[row.method] for row in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean command time (ms)")
    ax.set_title("Average Controller Computation Time", fontsize=15, fontweight="bold", pad=16)
    _add_group_labels(ax, rows, y)
    _style_axis(ax)

    legend_handles = [
        Patch(facecolor=_TIME_COLORS[method], label=_METHOD_LABELS[method])
        for method in dict.fromkeys(row.method for row in rows)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=6,
        frameon=False,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_plots(tuning_dir: Path, out_dir: Path, include_mppi: bool = False) -> list[Path]:
    methods = _METHODS + (_OPTIONAL_METHODS if include_mppi else ())
    rows = _read_rows(tuning_dir, methods)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "with_mppi" if include_mppi else "without_mppi"
    outcome_path = out_dir / f"benchmark_outcomes_{suffix}.png"
    time_path = out_dir / f"benchmark_command_time_{suffix}.png"
    _plot_outcomes(rows, outcome_path)
    _plot_command_time(rows, time_path)
    return [outcome_path, time_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary charts for tuned MPPI benchmark artifacts.")
    parser.add_argument("--tuning-dir", type=Path, default=Path("output/tuning"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/tuning/summary_charts"))
    parser.add_argument("--include-mppi", action="store_true", help="Include vanilla MPPI in the plots.")
    args = parser.parse_args()

    paths = generate_plots(args.tuning_dir, args.out_dir, include_mppi=args.include_mppi)
    print("Wrote:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
