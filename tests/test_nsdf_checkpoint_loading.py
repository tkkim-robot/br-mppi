from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import jax.numpy as jnp
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robots import create_robot
from sdf import (
    MobileArmSDF,
    PretrainedMobileArmSDF,
    PretrainedSDFIntegrityError,
    PretrainedSDFUnavailable,
    PretrainedShapeSDF,
    default_obstacle_field,
    load_pretrained_sdf_for_robot,
)
from sdf.nsdf_training import rectangle_signed_distance


def test_default_loader_remains_numerically_identical_to_explicit_legacy() -> None:
    default_model = load_pretrained_sdf_for_robot("unicycle")
    explicit_legacy = load_pretrained_sdf_for_robot("unicycle", variant="legacy")
    points = jnp.asarray(
        ((0.0, 0.0), (0.5, 0.2), (0.8, -0.4), (-1.1, 0.7)),
        dtype=jnp.float32,
    )
    field = default_obstacle_field("unicycle")
    state = create_robot("unicycle").default_state

    assert isinstance(default_model, PretrainedShapeSDF)
    assert isinstance(explicit_legacy, PretrainedShapeSDF)
    assert default_model.spec.variant == "legacy"
    assert default_model.geometry_padding == 0.0
    assert default_model.model_margin == 0.0
    assert jnp.array_equal(
        default_model.signed_distance(points),
        explicit_legacy.signed_distance(points),
    )
    assert jnp.array_equal(
        default_model.obstacle_barriers(state, field),
        explicit_legacy.obstacle_barriers(state, field),
    )


@pytest.mark.parametrize(
    ("robot_name", "width", "height", "near_band", "max_near_mae", "padding", "seed"),
    (
        ("unicycle", 1.0, 0.4, 0.04, 0.012, 0.08, 0),
        ("dynamic_unicycle", 1.0, 0.4, 0.04, 0.012, 0.08, 0),
        ("planar_quadrotor", 0.56, 0.28, 0.028, 0.009, 0.06, 2),
    ),
)
def test_retrained_rigid_checkpoint_is_accurate_and_exposes_provenance(
    robot_name: str,
    width: float,
    height: float,
    near_band: float,
    max_near_mae: float,
    padding: float,
    seed: int,
) -> None:
    model = load_pretrained_sdf_for_robot(robot_name, variant="retrained")
    assert isinstance(model, PretrainedShapeSDF)

    x = jnp.linspace(-1.4, 1.4, 101, dtype=jnp.float32)
    y = jnp.linspace(-1.2, 1.2, 91, dtype=jnp.float32)
    grid_x, grid_y = jnp.meshgrid(x, y)
    points = jnp.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    truth = rectangle_signed_distance(points, width=width, height=height)
    predictions = model.signed_distance(points)
    near_surface = jnp.abs(truth) <= near_band
    near_mae = jnp.mean(jnp.abs(predictions[near_surface] - truth[near_surface]))

    parameter_dtypes = {
        value.dtype
        for layer in model.params["params"].values()
        for value in layer.values()
    }
    assert parameter_dtypes == {jnp.dtype(jnp.float32)}
    assert float(near_mae) < max_near_mae
    assert model.spec.variant == "retrained"
    assert model.metadata is not None
    assert model.geometry_padding == pytest.approx(padding)
    assert model.model_margin == pytest.approx(
        model.metadata["validation"]["recommended_conservative_margin"]
    )
    assert model.provenance["training_seed"] == seed
    assert model.provenance["geometry_padding"] == pytest.approx(padding)
    assert model.provenance["model_margin"] == pytest.approx(model.model_margin)
    assert model.provenance["total_margin"] == pytest.approx(
        padding + model.model_margin
    )


def test_retrained_rigid_barrier_applies_padding_and_model_margin_separately() -> None:
    model = load_pretrained_sdf_for_robot("unicycle", variant="retrained")
    assert isinstance(model, PretrainedShapeSDF)
    unshifted = PretrainedShapeSDF(
        robot_name=model.robot_name,
        spec=model.spec,
        params=model.params,
        metadata=model.metadata,
    )
    field = default_obstacle_field("unicycle")
    state = create_robot("unicycle").default_state

    raw_barriers = unshifted.obstacle_barriers(state, field)
    conservative_barriers = model.obstacle_barriers(state, field)

    assert jnp.allclose(
        conservative_barriers,
        raw_barriers - model.geometry_padding - model.model_margin,
        atol=1e-6,
    )


def test_retrained_mobile_arm_loads_verified_compositional_models() -> None:
    model = load_pretrained_sdf_for_robot("mobile_arm", variant="retrained")

    assert isinstance(model, MobileArmSDF)
    assert isinstance(model, PretrainedMobileArmSDF)
    assert model.obstacle_mode == "narrow_surface"
    assert model.geometry_padding == pytest.approx(0.02)
    assert model.base_metadata is not None
    assert model.link_metadata is not None
    assert model.model_margin == 0.0
    assert model.base_model_margin == pytest.approx(
        model.base_metadata["validation"]["recommended_conservative_margin"]
    )
    assert model.link_model_margin == pytest.approx(
        model.link_metadata["validation"]["recommended_conservative_margin"]
    )
    assert model.points_per_obstacle == 64
    assert model.narrow_phase_points == 4
    assert model.broad_phase_parts == 2
    assert model.provenance["base"]["checkpoint"] == "mobile_arm_base_model_4_16.npy"
    assert len(model.provenance["base"]["checkpoint_sha256"]) == 64
    assert model.provenance["base"]["training_seed"] == 1
    assert model.provenance["link"]["training_seed"] == 0
    assert model.provenance["runtime"]["obstacle_mode"] == "narrow_surface"
    assert model.provenance["runtime"]["narrow_phase_points"] == 4
    assert model.provenance["runtime"]["broad_phase_parts"] == 2
    assert model.provenance["runtime"]["learned_blend_full_distance"] == pytest.approx(
        0.05
    )
    assert model.provenance["runtime"]["learned_blend_zero_distance"] == pytest.approx(
        0.20
    )
    assert "base_model_margin" in model.description
    assert model.base_metadata["geometry"]["width"] == pytest.approx(1.5)
    assert model.link_metadata["geometry"]["width"] == pytest.approx(0.8)

    robot = create_robot("mobile_arm")
    field = default_obstacle_field("mobile_arm")
    barriers = model.obstacle_barriers(robot.default_state, field)
    assert barriers.shape == (len(field.obstacles),)
    assert bool(jnp.all(jnp.isfinite(barriers)))


def test_legacy_mobile_arm_remains_explicitly_unavailable() -> None:
    with pytest.raises(PretrainedSDFUnavailable, match="canonical single-shape"):
        load_pretrained_sdf_for_robot("mobile_arm")


def test_retrained_loader_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source_dir = REPO_ROOT / "sdf" / "trained_models"
    target_dir = tmp_path / "sdf" / "trained_models"
    target_dir.mkdir(parents=True)
    checkpoint_name = "link1_retrained_model_4_16.npy"
    metadata_name = "link1_retrained_model_4_16.json"
    shutil.copy2(source_dir / checkpoint_name, target_dir / checkpoint_name)
    shutil.copy2(source_dir / metadata_name, target_dir / metadata_name)
    checkpoint = target_dir / checkpoint_name
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 0x01
    checkpoint.write_bytes(payload)

    with pytest.raises(PretrainedSDFIntegrityError, match="SHA-256 mismatch"):
        load_pretrained_sdf_for_robot(
            "unicycle",
            variant="retrained",
            repo_root=tmp_path,
        )


def test_retrained_loader_requires_code_pinned_metadata_checksum(tmp_path: Path) -> None:
    source_dir = REPO_ROOT / "sdf" / "trained_models"
    target_dir = tmp_path / "sdf" / "trained_models"
    target_dir.mkdir(parents=True)
    checkpoint_name = "link1_retrained_model_4_16.npy"
    metadata_name = "link1_retrained_model_4_16.json"
    shutil.copy2(source_dir / checkpoint_name, target_dir / checkpoint_name)
    shutil.copy2(source_dir / metadata_name, target_dir / metadata_name)
    metadata_path = target_dir / metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint"]["sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PretrainedSDFIntegrityError, match="trusted pin"):
        load_pretrained_sdf_for_robot(
            "unicycle",
            variant="retrained",
            repo_root=tmp_path,
        )


def test_retrained_loader_requires_exact_architecture_order(tmp_path: Path) -> None:
    source_dir = REPO_ROOT / "sdf" / "trained_models"
    target_dir = tmp_path / "sdf" / "trained_models"
    target_dir.mkdir(parents=True)
    checkpoint_name = "link1_retrained_model_4_16.npy"
    metadata_name = "link1_retrained_model_4_16.json"
    shutil.copy2(source_dir / checkpoint_name, target_dir / checkpoint_name)
    shutil.copy2(source_dir / metadata_name, target_dir / metadata_name)
    metadata_path = target_dir / metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["architecture"]["layer_order"] = list(
        reversed(metadata["architecture"]["layer_order"])
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PretrainedSDFIntegrityError, match="architecture"):
        load_pretrained_sdf_for_robot(
            "unicycle",
            variant="retrained",
            repo_root=tmp_path,
        )


def test_retrained_loader_requires_exact_architecture_width(tmp_path: Path) -> None:
    source_dir = REPO_ROOT / "sdf" / "trained_models"
    target_dir = tmp_path / "sdf" / "trained_models"
    target_dir.mkdir(parents=True)
    checkpoint_name = "link1_retrained_model_4_16.npy"
    metadata_name = "link1_retrained_model_4_16.json"
    shutil.copy2(source_dir / checkpoint_name, target_dir / checkpoint_name)
    shutil.copy2(source_dir / metadata_name, target_dir / metadata_name)
    metadata_path = target_dir / metadata_name
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["architecture"]["hidden_dim"] = 8
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PretrainedSDFIntegrityError, match="architecture"):
        load_pretrained_sdf_for_robot(
            "unicycle",
            variant="retrained",
            repo_root=tmp_path,
        )


def test_loader_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="variant"):
        load_pretrained_sdf_for_robot("unicycle", variant="future")  # type: ignore[arg-type]
