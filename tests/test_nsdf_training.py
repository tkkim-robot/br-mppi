from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdf.nsdf_network import (
    init_legacy_sdf_params,
    legacy_sdf_values,
    legacy_sdf_values_and_gradients,
    load_legacy_checkpoint,
    validate_legacy_params,
)
from sdf.nsdf_training import (
    RectangleSpec,
    TrainingConfig,
    TrainingResult,
    evaluate_rectangle_contact_shell,
    evaluate_rectangle_sdf,
    rectangle_signed_distance,
    sample_rectangle_training_batch,
    save_training_artifacts,
    train_rectangle_sdf,
)
import sdf.nsdf_training as nsdf_training_module
from sdf.pretrained_sdf import load_pretrained_sdf_for_robot


def test_rectangle_signed_distance_matches_axis_aligned_geometry() -> None:
    points = jnp.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (0.0, 0.2),
            (0.6, 0.0),
            (0.6, 0.3),
        ),
        dtype=jnp.float32,
    )

    distances = rectangle_signed_distance(points, width=1.0, height=0.4)

    expected = jnp.asarray(
        (-0.2, 0.0, 0.0, 0.1, jnp.sqrt(0.02)),
        dtype=jnp.float32,
    )
    assert jnp.allclose(distances, expected, atol=1e-6)


def test_new_network_evaluator_matches_deployed_legacy_evaluator() -> None:
    deployed = load_pretrained_sdf_for_robot("unicycle")
    points = jnp.asarray(
        ((0.0, 0.0), (0.37, -0.16), (0.8, 0.2), (-1.1, 0.9)),
        dtype=jnp.float32,
    )

    shared_values, shared_gradients = legacy_sdf_values_and_gradients(deployed.params, points)
    deployed_values, deployed_gradients = deployed._value_and_local_gradient(
        jnp.column_stack((points, jnp.zeros((points.shape[0],), dtype=jnp.float32)))
    )

    assert jnp.array_equal(shared_values, deployed_values)
    assert jnp.array_equal(shared_gradients, deployed_gradients)


def test_legacy_architecture_matches_independent_golden_values_and_gradients() -> None:
    params = load_legacy_checkpoint(
        REPO_ROOT / "sdf" / "trained_models" / "link1_model_4_16.npy"
    )
    points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.37, -0.16, 0.0),
            (0.8, 0.2, 0.0),
            (-1.1, 0.9, 0.0),
        ),
        dtype=jnp.float32,
    )
    expected_values = jnp.asarray(
        (-0.1190254092, 0.002252370119, 0.3051026762, 0.9213404655),
        dtype=jnp.float32,
    )
    expected_gradients = jnp.asarray(
        (
            (0.03600905836, -0.007720768452, -0.9218069315),
            (0.366820842, -0.5537539124, -0.6502620578),
            (0.8953502178, 0.2818189859, -0.3868741691),
            (-0.6172310114, 0.7531374693, -0.136574924),
        ),
        dtype=jnp.float32,
    )

    with jax.default_matmul_precision("highest"):
        values, gradients = legacy_sdf_values_and_gradients(params, points)

    assert jnp.allclose(values, expected_values, rtol=2e-6, atol=2e-6)
    assert jnp.allclose(gradients, expected_gradients, rtol=2e-6, atol=2e-6)


def test_initializer_and_sampler_are_deterministic_float32() -> None:
    rectangle = RectangleSpec("test_rectangle", width=1.0, height=0.4)
    config = TrainingConfig(
        steps=1,
        batch_size=30,
        eikonal_batch_size=10,
        validation_points=30,
        boundary_validation_points=8,
        contact_shell_points=8,
    )
    key = jax.random.PRNGKey(17)

    params_a = init_legacy_sdf_params(key)
    params_b = init_legacy_sdf_params(key)
    batch_a = sample_rectangle_training_batch(key, rectangle, config)
    batch_b = sample_rectangle_training_batch(key, rectangle, config)

    assert validate_legacy_params(params_a) == 16
    for leaf_a, leaf_b in zip(
        jax.tree.leaves(params_a),
        jax.tree.leaves(params_b),
        strict=True,
    ):
        assert leaf_a.dtype == jnp.float32
        assert jnp.array_equal(leaf_a, leaf_b)
    for array_a, array_b in zip(batch_a, batch_b, strict=True):
        assert array_a.dtype == jnp.float32
        assert jnp.array_equal(array_a, array_b)
    # One third of the supervised batch is sampled exactly on the boundary.
    assert int(jnp.count_nonzero(jnp.abs(batch_a[1]) < 1e-7)) >= 10


def test_validation_seed_is_independent_of_training_seed() -> None:
    assert TrainingConfig(seed=0).validation_seed == 10_000
    assert TrainingConfig(seed=99).validation_seed == 10_000


def test_parameter_validation_rejects_wrong_dtype_and_nonfinite_values() -> None:
    params = init_legacy_sdf_params(jax.random.PRNGKey(0))
    wrong_dtype = copy.deepcopy(params)
    wrong_dtype["params"]["Dense_0"]["kernel"] = np.asarray(
        wrong_dtype["params"]["Dense_0"]["kernel"],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="float32"):
        validate_legacy_params(wrong_dtype)

    nonfinite = copy.deepcopy(params)
    nonfinite["params"]["Dense_1"]["bias"] = nonfinite["params"]["Dense_1"][
        "bias"
    ].at[0].set(jnp.nan)
    with pytest.raises(ValueError, match="finite"):
        validate_legacy_params(nonfinite)

    with pytest.raises(ValueError, match="hidden_dim"):
        validate_legacy_params(
            init_legacy_sdf_params(jax.random.PRNGKey(1), hidden_dim=8),
            expected_hidden_dim=16,
        )

    extra_parameter = copy.deepcopy(params)
    extra_parameter["params"]["Dense_3"]["unexpected"] = np.zeros(
        (1,),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="exactly"):
        validate_legacy_params(extra_parameter)


def test_training_config_requires_integer_hidden_width() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        TrainingConfig(hidden_dim=16.0)  # type: ignore[arg-type]


def test_short_training_run_is_finite_and_reports_safety_metrics() -> None:
    rectangle = RectangleSpec("test_rectangle", width=1.0, height=0.4)
    config = TrainingConfig(
        seed=3,
        steps=2,
        batch_size=12,
        eikonal_batch_size=6,
        log_every=1,
        validation_points=60,
        boundary_validation_points=12,
        contact_shell_points=8,
    )

    result = train_rectangle_sdf(rectangle, config)

    assert len(result.loss_history) == 2
    assert all(
        np.asarray(jax.device_get(parameter)).dtype == np.dtype(np.float32)
        for parameter in jax.tree.leaves(result.params)
    )
    assert all(
        np.isfinite(value)
        for record in result.loss_history
        for key, value in record.items()
        if key != "step"
    )
    assert result.validation_metrics["false_safe_rate"] >= 0.0
    assert result.validation_metrics["recommended_conservative_margin"] >= 0.0
    assert result.validation_metrics["recommended_conservative_margin"] == pytest.approx(
        max(
            result.validation_metrics["near_surface_max_overestimation"],
            result.validation_metrics["contact_shell"]["max_overestimation"],
        )
    )


def test_checkpoint_and_metadata_round_trip_without_overwrite(tmp_path) -> None:
    rectangle = RectangleSpec("mobile_arm_base", width=1.5, height=1.0)
    config = TrainingConfig(
        steps=1,
        batch_size=6,
        eikonal_batch_size=2,
        validation_points=6,
        boundary_validation_points=4,
        contact_shell_points=8,
    )
    params = init_legacy_sdf_params(jax.random.PRNGKey(0))
    validation = evaluate_rectangle_sdf(
        params,
        rectangle,
        seed=4,
        num_points=12,
        boundary_points=8,
        contact_shell_points=8,
    )
    result = TrainingResult(
        rectangle=rectangle,
        config=config,
        params=params,
        loss_history=(),
        validation_metrics=validation,
    )
    checkpoint_path = tmp_path / "mobile_arm_base_model_4_16.npy"

    checkpoint, metadata = save_training_artifacts(
        result,
        checkpoint_path,
        source_revision="test-revision",
        source_provenance={
            "git_dirty": True,
            "source_file_sha256": {"sdf/nsdf_network.py": "a" * 64},
        },
    )

    loaded = load_legacy_checkpoint(checkpoint)
    probe = jnp.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=jnp.float32)
    assert jnp.array_equal(legacy_sdf_values(params, probe), legacy_sdf_values(loaded, probe))
    document = json.loads(metadata.read_text(encoding="utf-8"))
    assert document["format_version"] == 1
    assert document["geometry"]["width"] == 1.5
    assert document["geometry"]["height"] == 1.0
    assert document["architecture"]["dtype"] == "float32"
    assert document["validation"]["recommended_conservative_margin"] >= 0.0
    assert document["margin_scope"]["type"] == "empirical_sampled"
    assert "validation.collision_max_overestimation" in document["margin_scope"]["excludes"]
    assert document["provenance"]["source"]["git_dirty"]
    assert (
        document["provenance"]["source"]["source_file_sha256"][
            "sdf/nsdf_network.py"
        ]
        == "a" * 64
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_training_artifacts(result, checkpoint_path)


def test_saver_rejects_config_parameter_width_mismatch(tmp_path) -> None:
    rectangle = RectangleSpec("mismatch", width=1.0, height=0.4)
    config = TrainingConfig(
        hidden_dim=16,
        steps=1,
        batch_size=6,
        eikonal_batch_size=2,
        validation_points=6,
        boundary_validation_points=4,
        contact_shell_points=8,
    )
    result = TrainingResult(
        rectangle=rectangle,
        config=config,
        params=init_legacy_sdf_params(jax.random.PRNGKey(0), hidden_dim=8),
        loss_history=(),
        validation_metrics={},
    )
    checkpoint = tmp_path / "mismatch.npy"

    with pytest.raises(ValueError, match="hidden_dim"):
        save_training_artifacts(result, checkpoint)

    assert not checkpoint.exists()
    assert not checkpoint.with_suffix(".json").exists()


def test_failed_checkpoint_write_removes_exclusive_partial_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangle = RectangleSpec("write_failure", width=1.0, height=0.4)
    config = TrainingConfig(
        steps=1,
        batch_size=6,
        eikonal_batch_size=2,
        validation_points=6,
        boundary_validation_points=4,
        contact_shell_points=8,
    )
    result = TrainingResult(
        rectangle=rectangle,
        config=config,
        params=init_legacy_sdf_params(jax.random.PRNGKey(0)),
        loss_history=(),
        validation_metrics={},
    )
    checkpoint = tmp_path / "partial.npy"

    def fail_after_writing(file_handle, *_args, **_kwargs):
        file_handle.write(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(nsdf_training_module.np, "save", fail_after_writing)
    with pytest.raises(OSError, match="simulated"):
        save_training_artifacts(result, checkpoint)

    assert not checkpoint.exists()
    assert not checkpoint.with_suffix(".json").exists()


def test_contact_shell_covers_scaled_link_and_large_circle_queries() -> None:
    params = init_legacy_sdf_params(jax.random.PRNGKey(5))
    rectangle = RectangleSpec("mobile_arm_canonical_link", width=0.8, height=0.2)

    metrics = evaluate_rectangle_contact_shell(
        params,
        rectangle,
        scales=(1.0, 0.25),
        radii=(0.28, 0.62),
        points_per_shell=16,
        seed=9,
    )

    assert len(metrics["cases"]) == 4
    smallest_link_large_circle = next(
        case
        for case in metrics["cases"]
        if case["scale"] == 0.25 and case["radius"] == 0.62
    )
    assert smallest_link_large_circle["canonical_exact_distance"] == pytest.approx(2.48)
    assert all(case["exact_contact_max_abs_error"] < 1e-5 for case in metrics["cases"])
    assert metrics["max_overestimation"] == max(
        case["max_overestimation"] for case in metrics["cases"]
    )
