from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (REPO_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import basic_demo
import random_benchmark
import sanity_check
import train_nsdf
from sdf import MobileArmSDF


@pytest.mark.parametrize(
    "module",
    (basic_demo, random_benchmark, sanity_check),
)
def test_nsdf_cli_defaults_to_analytic_with_retrained_variant_available(
    module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(module.__file__)])

    args = module.parse_args()

    assert not args.nsdf
    assert args.nsdf_variant == "retrained"
    assert args.mobile_nsdf_points == 64
    assert args.mobile_nsdf_narrow_points == 4
    assert args.mobile_nsdf_analytic_guard


def test_demo_and_benchmark_load_retrained_mobile_arm() -> None:
    demo_model = basic_demo.load_selected_sdf(
        "mobile_arm",
        "brmppi",
        nsdf=True,
        variant="retrained",
        mobile_points=32,
        mobile_narrow_points=3,
    )
    benchmark_model = random_benchmark.load_benchmark_sdf(
        "mobile_arm",
        nsdf=True,
        variant="retrained",
        mobile_points=32,
        mobile_narrow_points=3,
        mobile_analytic_guard=False,
    )

    assert isinstance(demo_model, MobileArmSDF)
    assert demo_model.obstacle_mode == "narrow_surface"
    assert demo_model.points_per_obstacle == 32
    assert demo_model.narrow_phase_points == 3
    assert isinstance(benchmark_model, MobileArmSDF)
    assert not benchmark_model.analytic_guard


def test_programmatic_loader_default_preserves_legacy_checkpoint() -> None:
    model = random_benchmark.load_benchmark_sdf("unicycle", nsdf=True)

    assert model.spec.variant == "legacy"


def test_cli_rejects_nsdf_for_non_brmppi_and_unsupported_geometry() -> None:
    with pytest.raises(NotImplementedError, match="only implemented for brmppi"):
        basic_demo.load_selected_sdf(
            "unicycle",
            "mppi",
            nsdf=True,
            variant="retrained",
        )
    with pytest.raises(NotImplementedError, match="single_integrator"):
        random_benchmark.load_benchmark_sdf(
            "single_integrator",
            nsdf=True,
            variant="retrained",
        )


def test_output_filename_distinguishes_checkpoint_variant() -> None:
    path = random_benchmark.default_output_path(
        "unicycle",
        ("brmppi",),
        barrier_source="neural_sdf",
        nsdf_variant="retrained",
        tuned_config=True,
    )

    assert "neural_sdf_retrained" in path.name


def test_runtime_provenance_is_machine_readable() -> None:
    provenance = random_benchmark.runtime_provenance()

    assert provenance["git_revision"]
    assert provenance["git_dirty"] in {True, False, None}
    assert provenance["jax_version"]
    assert provenance["jax_backend"] in {"cpu", "gpu", "tpu"}
    assert provenance["jax_devices"]
    json.dumps(provenance)


def test_benchmark_json_encoder_handles_provenance_values() -> None:
    encoded = json.dumps(
        {
            "path": Path("checkpoint.npy"),
            "scalar": np.float32(0.25),
            "array": np.asarray([1, 2], dtype=np.int32),
        },
        default=random_benchmark.json_default,
    )

    assert json.loads(encoded) == {
        "path": "checkpoint.npy",
        "scalar": 0.25,
        "array": [1, 2],
    }


def test_training_cli_uses_fixed_validation_seed_and_hashes_sources() -> None:
    args = train_nsdf.parse_args([])
    provenance = train_nsdf._source_provenance()

    assert args.validation_seed == 10_000
    assert provenance["git_revision"]
    assert provenance["git_dirty"] in {True, False, None}
    source_hashes = provenance["source_file_sha256"]
    assert set(source_hashes) == set(train_nsdf.TRAINING_SOURCE_FILES)
    assert all(len(checksum) == 64 for checksum in source_hashes.values())
