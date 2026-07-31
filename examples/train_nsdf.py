from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdf.nsdf_network import load_legacy_checkpoint
from sdf.nsdf_training import (
    RectangleSpec,
    TrainingConfig,
    evaluate_rectangle_sdf,
    save_training_artifacts,
    train_rectangle_sdf,
)


RECTANGLE_PRESETS = {
    "link1": RectangleSpec("link1_rectangle", width=1.0, height=0.4),
    "link7": RectangleSpec("link7_quadrotor_rectangle", width=0.56, height=0.28),
    "mobile_base": RectangleSpec("mobile_arm_base", width=1.5, height=1.0),
    "mobile_link": RectangleSpec("mobile_arm_canonical_link", width=0.8, height=0.2),
}
TRAINING_SOURCE_FILES = (
    "sdf/nsdf_network.py",
    "sdf/nsdf_training.py",
    "examples/train_nsdf.py",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a deterministic float32 rectangle NSDF compatible with the "
            "repository's legacy 4x16 .npy checkpoints."
        )
    )
    parser.add_argument("--preset", choices=tuple(RECTANGLE_PRESETS), default="link1")
    parser.add_argument("--width", type=float, help="custom rectangle width in meters")
    parser.add_argument("--height", type=float, help="custom rectangle height in meters")
    parser.add_argument("--name", help="custom geometry name stored in metadata")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=10_000,
        help="fixed validation seed shared across training seeds",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1_536)
    parser.add_argument("--eikonal-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eikonal-weight", type=float, default=0.01)
    parser.add_argument("--domain-padding", type=float, default=1.0)
    parser.add_argument("--near-surface-band", type=float)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--validation-points", type=int, default=24_000)
    parser.add_argument("--boundary-validation-points", type=int, default=4_096)
    parser.add_argument(
        "--contact-shell-scales",
        type=float,
        nargs="+",
        help="rectangle scales checked by exact-contact circle-center validation",
    )
    parser.add_argument(
        "--contact-shell-radii",
        type=float,
        nargs="+",
        default=(0.28, 0.62),
        help="circle radii checked by exact-contact validation",
    )
    parser.add_argument("--contact-shell-points", type=int, default=2_048)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "new .npy checkpoint path; defaults to "
            "output/nsdf_training/<preset>_model_4x<hidden>_seed<seed>.npy"
        ),
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        help="optional legacy checkpoint evaluated on the same deterministic validation set",
    )
    return parser.parse_args(argv)


def resolve_rectangle(args: argparse.Namespace) -> RectangleSpec:
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be specified together")
    if args.width is None:
        preset = RECTANGLE_PRESETS[args.preset]
        return RectangleSpec(
            name=args.name or preset.name,
            width=preset.width,
            height=preset.height,
            units=preset.units,
        )
    return RectangleSpec(
        name=args.name or "custom_rectangle",
        width=args.width,
        height=args.height,
    )


def default_output_path(args: argparse.Namespace) -> Path:
    filename = (
        f"{args.preset}_model_4x{args.hidden_dim}_seed{args.seed}.npy"
        if args.width is None
        else f"custom_rectangle_model_4x{args.hidden_dim}_seed{args.seed}.npy"
    )
    return REPO_ROOT / "output" / "nsdf_training" / filename


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rectangle = resolve_rectangle(args)
    contact_shell_scales = (
        tuple(args.contact_shell_scales)
        if args.contact_shell_scales is not None
        else ((1.0, 0.75, 0.5, 0.25) if args.preset == "mobile_link" else (1.0,))
    )
    config = TrainingConfig(
        seed=args.seed,
        validation_seed=args.validation_seed,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        batch_size=args.batch_size,
        eikonal_batch_size=args.eikonal_batch_size,
        learning_rate=args.learning_rate,
        eikonal_weight=args.eikonal_weight,
        domain_padding=args.domain_padding,
        near_surface_band=args.near_surface_band,
        log_every=args.log_every,
        validation_points=args.validation_points,
        boundary_validation_points=args.boundary_validation_points,
        contact_shell_scales=contact_shell_scales,
        contact_shell_radii=tuple(args.contact_shell_radii),
        contact_shell_points=args.contact_shell_points,
    )
    output_path = (args.output or default_output_path(args)).resolve()
    source_before_training = _source_provenance()

    print(
        f"Training {rectangle.name}: {rectangle.width:g} x {rectangle.height:g} "
        f"{rectangle.units}, seed={config.seed}, steps={config.steps}"
    )

    def print_progress(record: dict[str, float | int]) -> None:
        print(
            f"step={record['step']:>6} "
            f"loss={record['loss']:.6g} "
            f"distance_mse={record['distance_mse']:.6g} "
            f"eikonal={record['eikonal_loss']:.6g} "
            f"lr={record['learning_rate']:.3g}"
        )

    result = train_rectangle_sdf(
        rectangle,
        config,
        progress_callback=print_progress,
    )

    comparison = None
    if args.reference_checkpoint is not None:
        reference_params = load_legacy_checkpoint(args.reference_checkpoint)
        reference_metrics = evaluate_rectangle_sdf(
            reference_params,
            rectangle,
            seed=config.validation_seed,
            num_points=config.validation_points,
            boundary_points=config.boundary_validation_points,
            domain_padding=config.domain_padding,
            near_surface_band=config.near_surface_band,
            contact_shell_scales=config.contact_shell_scales,
            contact_shell_radii=config.contact_shell_radii,
            contact_shell_points=config.contact_shell_points,
        )
        comparison = {
            "reference_checkpoint": str(args.reference_checkpoint.resolve()),
            "reference_validation": reference_metrics,
            "trained_minus_reference": {
                key: result.validation_metrics[key] - reference_metrics[key]
                for key in (
                    "distance_mae",
                    "near_surface_mae",
                    "false_safe_rate",
                    "boundary_mae",
                    "gradient_l2_error_mean",
                    "eikonal_abs_error_mean",
                )
            },
        }

    source_after_training = _source_provenance()
    source_provenance = {
        "before_training": source_before_training,
        "after_training": source_after_training,
        "changed_during_training": (
            source_before_training != source_after_training
        ),
    }
    checkpoint, metadata = save_training_artifacts(
        result,
        output_path,
        source_revision=source_before_training["git_revision"],
        source_provenance=source_provenance,
        extra_metadata=comparison,
    )
    print("Validation metrics:")
    print(json.dumps(result.validation_metrics, indent=2, sort_keys=True))
    if comparison is not None:
        print("Reference comparison:")
        print(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"Checkpoint: {checkpoint}")
    print(f"Metadata:   {metadata}")
    return 0


def _git_revision() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _source_provenance() -> dict[str, object]:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    status_lines = status.stdout.splitlines() if status.returncode == 0 else []
    return {
        "git_revision": _git_revision(),
        "git_dirty": bool(status_lines) if status.returncode == 0 else None,
        "git_status": status_lines,
        "source_file_sha256": {
            relative_path: _sha256(REPO_ROOT / relative_path)
            for relative_path in TRAINING_SOURCE_FILES
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
