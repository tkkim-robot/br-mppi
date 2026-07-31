from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from sdf.nsdf_network import (
    LEGACY_ARCHITECTURE_ORDER,
    NSDF_DTYPE,
    init_legacy_sdf_params,
    legacy_sdf_values,
    legacy_sdf_values_and_gradients,
    params_to_numpy,
    validate_legacy_params,
)


@dataclass(frozen=True)
class RectangleSpec:
    name: str
    width: float
    height: float
    units: str = "m"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("rectangle name must not be empty")
        if not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("rectangle width must be finite and positive")
        if not np.isfinite(self.height) or self.height <= 0.0:
            raise ValueError("rectangle height must be finite and positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    validation_seed: int = 10_000
    hidden_dim: int = 16
    steps: int = 20_000
    batch_size: int = 1_536
    eikonal_batch_size: int = 512
    learning_rate: float = 1e-3
    min_learning_rate_ratio: float = 0.1
    eikonal_weight: float = 0.01
    domain_padding: float = 1.0
    near_surface_band: float | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_clip_norm: float = 10.0
    log_every: int = 250
    validation_points: int = 24_000
    boundary_validation_points: int = 4_096
    contact_shell_scales: tuple[float, ...] = (1.0,)
    contact_shell_radii: tuple[float, ...] = (0.28, 0.62)
    contact_shell_points: int = 2_048

    def __post_init__(self) -> None:
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if (
            not isinstance(self.validation_seed, (int, np.integer))
            or self.validation_seed < 0
        ):
            raise ValueError("validation_seed must be a nonnegative integer")
        if (
            not isinstance(self.hidden_dim, (int, np.integer))
            or self.hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.batch_size < 6:
            raise ValueError("batch_size must be at least 6")
        if self.eikonal_batch_size < 2:
            raise ValueError("eikonal_batch_size must be at least 2")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if self.eikonal_weight < 0.0:
            raise ValueError("eikonal_weight must be nonnegative")
        if self.domain_padding <= 0.0:
            raise ValueError("domain_padding must be positive")
        if self.near_surface_band is not None and self.near_surface_band <= 0.0:
            raise ValueError("near_surface_band must be positive when specified")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.adam_epsilon <= 0.0:
            raise ValueError("adam_epsilon must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.validation_points < 6:
            raise ValueError("validation_points must be at least 6")
        if self.boundary_validation_points < 4:
            raise ValueError("boundary_validation_points must be at least 4")
        if not self.contact_shell_scales or any(
            not np.isfinite(scale) or scale <= 0.0 for scale in self.contact_shell_scales
        ):
            raise ValueError("contact_shell_scales must contain positive finite values")
        if not self.contact_shell_radii or any(
            not np.isfinite(radius) or radius <= 0.0 for radius in self.contact_shell_radii
        ):
            raise ValueError("contact_shell_radii must contain positive finite values")
        if self.contact_shell_points < 8:
            raise ValueError("contact_shell_points must be at least 8")


@dataclass(frozen=True)
class TrainingResult:
    rectangle: RectangleSpec
    config: TrainingConfig
    params: Mapping[str, Any]
    loss_history: tuple[dict[str, float | int], ...]
    validation_metrics: dict[str, Any]


ProgressCallback = Callable[[dict[str, float | int]], None]


def rectangle_signed_distance(
    points: jax.Array,
    *,
    width: float,
    height: float,
) -> jax.Array:
    """Exact signed distance to an axis-aligned rectangle centered at zero."""
    xy = _points_to_xy(points)
    half_extents = jnp.asarray((0.5 * width, 0.5 * height), dtype=NSDF_DTYPE)
    offset = jnp.abs(xy) - half_extents
    outside = jnp.linalg.norm(jnp.maximum(offset, 0.0), axis=1)
    inside = jnp.minimum(jnp.maximum(offset[:, 0], offset[:, 1]), 0.0)
    return outside + inside


def rectangle_signed_distance_and_gradient(
    points: jax.Array,
    *,
    width: float,
    height: float,
) -> tuple[jax.Array, jax.Array]:
    """Return exact rectangle distances and an XY subgradient."""
    xy = _points_to_xy(points)
    half_extents = jnp.asarray((0.5 * width, 0.5 * height), dtype=NSDF_DTYPE)
    offset = jnp.abs(xy) - half_extents
    positive_offset = jnp.maximum(offset, 0.0)
    outside_norm = jnp.linalg.norm(positive_offset, axis=1)
    distances = outside_norm + jnp.minimum(jnp.maximum(offset[:, 0], offset[:, 1]), 0.0)

    signs = jnp.sign(xy)
    outside_gradients = signs * positive_offset / jnp.maximum(outside_norm[:, None], 1e-8)
    x_is_nearest = offset[:, 0] >= offset[:, 1]
    inside_gradients = jnp.column_stack(
        (
            jnp.where(x_is_nearest, signs[:, 0], 0.0),
            jnp.where(x_is_nearest, 0.0, signs[:, 1]),
        )
    )
    gradients = jnp.where((outside_norm > 1e-8)[:, None], outside_gradients, inside_gradients)
    return distances, gradients


def sample_rectangle_training_batch(
    key: jax.Array,
    rectangle: RectangleSpec,
    config: TrainingConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample balanced uniform, near-surface, and boundary supervision."""
    near_band = _resolved_near_band(rectangle, config.near_surface_band)
    half_extents = jnp.asarray(
        (0.5 * rectangle.width, 0.5 * rectangle.height),
        dtype=NSDF_DTYPE,
    )
    domain_half_extents = half_extents + jnp.asarray(config.domain_padding, dtype=NSDF_DTYPE)
    batch_key, eikonal_key = jax.random.split(key)
    points = _sample_balanced_points(
        batch_key,
        config.batch_size,
        half_extents,
        domain_half_extents,
        near_band,
    )
    distances = rectangle_signed_distance(
        points,
        width=rectangle.width,
        height=rectangle.height,
    )
    eikonal_points = _sample_eikonal_points(
        eikonal_key,
        config.eikonal_batch_size,
        half_extents,
        domain_half_extents,
        near_band,
    )
    return points, distances, eikonal_points


def train_rectangle_sdf(
    rectangle: RectangleSpec,
    config: TrainingConfig = TrainingConfig(),
    *,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train a legacy-compatible rectangle NSDF deterministically."""
    init_key = jax.random.PRNGKey(config.seed)
    params = init_legacy_sdf_params(init_key, hidden_dim=config.hidden_dim)
    first_moment = jax.tree.map(jnp.zeros_like, params)
    second_moment = jax.tree.map(jnp.zeros_like, params)
    optimizer_step = jnp.asarray(0, dtype=jnp.int32)

    @jax.jit
    def training_step(
        current_params: Mapping[str, Any],
        current_first_moment: Mapping[str, Any],
        current_second_moment: Mapping[str, Any],
        current_step: jax.Array,
        sample_key: jax.Array,
    ) -> tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        jax.Array,
        dict[str, jax.Array],
    ]:
        points, targets, eikonal_points = sample_rectangle_training_batch(
            sample_key,
            rectangle,
            config,
        )

        def loss_function(
            candidate_params: Mapping[str, Any],
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            one = jnp.asarray(1.0, dtype=NSDF_DTYPE)
            eikonal_weight = jnp.asarray(config.eikonal_weight, dtype=NSDF_DTYPE)
            predictions = legacy_sdf_values(candidate_params, points)
            distance_mse = jnp.mean(jnp.square(predictions - targets))
            _, input_gradients = legacy_sdf_values_and_gradients(
                candidate_params,
                eikonal_points,
            )
            gradient_norms = jnp.linalg.norm(input_gradients[:, :2], axis=1)
            eikonal_loss = jnp.mean(jnp.square(gradient_norms - one))
            total_loss = distance_mse + eikonal_weight * eikonal_loss
            return total_loss, {
                "loss": total_loss,
                "distance_mse": distance_mse,
                "eikonal_loss": eikonal_loss,
            }

        (loss, losses), gradients = jax.value_and_grad(loss_function, has_aux=True)(current_params)
        del loss
        gradients, gradient_norm = _clip_gradients(gradients, config.gradient_clip_norm)
        next_step = current_step + 1
        learning_rate = _cosine_learning_rate(next_step, config)
        next_params, next_first_moment, next_second_moment = _adam_update(
            current_params,
            gradients,
            current_first_moment,
            current_second_moment,
            next_step,
            learning_rate,
            config,
        )
        losses["gradient_norm"] = gradient_norm
        losses["learning_rate"] = learning_rate
        return (
            next_params,
            next_first_moment,
            next_second_moment,
            next_step,
            losses,
        )

    history: list[dict[str, float | int]] = []
    training_key = jax.random.fold_in(jax.random.PRNGKey(config.seed), 1)
    for python_step in range(1, config.steps + 1):
        training_key, sample_key = jax.random.split(training_key)
        params, first_moment, second_moment, optimizer_step, losses = training_step(
            params,
            first_moment,
            second_moment,
            optimizer_step,
            sample_key,
        )
        should_record = (
            python_step == 1
            or python_step == config.steps
            or python_step % config.log_every == 0
        )
        if should_record:
            record: dict[str, float | int] = {"step": python_step}
            record.update(
                {
                    name: float(np.asarray(jax.device_get(value)))
                    for name, value in losses.items()
                }
            )
            history.append(record)
            if progress_callback is not None:
                progress_callback(record.copy())

    metrics = evaluate_rectangle_sdf(
        params,
        rectangle,
        seed=config.validation_seed,
        num_points=config.validation_points,
        boundary_points=config.boundary_validation_points,
        domain_padding=config.domain_padding,
        near_surface_band=_resolved_near_band(rectangle, config.near_surface_band),
        contact_shell_scales=config.contact_shell_scales,
        contact_shell_radii=config.contact_shell_radii,
        contact_shell_points=config.contact_shell_points,
    )
    return TrainingResult(
        rectangle=rectangle,
        config=config,
        params=params,
        loss_history=tuple(history),
        validation_metrics=metrics,
    )


def evaluate_rectangle_sdf(
    params: Mapping[str, Any],
    rectangle: RectangleSpec,
    *,
    seed: int = 10_000,
    num_points: int = 24_000,
    boundary_points: int = 4_096,
    domain_padding: float = 1.0,
    near_surface_band: float | None = None,
    contact_shell_scales: tuple[float, ...] = (1.0,),
    contact_shell_radii: tuple[float, ...] = (0.28, 0.62),
    contact_shell_points: int = 2_048,
) -> dict[str, Any]:
    """Evaluate distance, sign, boundary, and gradient quality."""
    validate_legacy_params(params)
    if num_points < 6 or boundary_points < 4:
        raise ValueError("num_points must be >= 6 and boundary_points must be >= 4")
    if domain_padding <= 0.0:
        raise ValueError("domain_padding must be positive")
    near_band = _resolved_near_band(rectangle, near_surface_band)
    half_extents = jnp.asarray(
        (0.5 * rectangle.width, 0.5 * rectangle.height),
        dtype=NSDF_DTYPE,
    )
    domain_half_extents = half_extents + jnp.asarray(domain_padding, dtype=NSDF_DTYPE)
    points_key, boundary_key = jax.random.split(jax.random.PRNGKey(seed))
    points = _sample_balanced_points(
        points_key,
        num_points,
        half_extents,
        domain_half_extents,
        near_band,
    )
    boundary_xy, _ = _sample_rectangle_boundary(boundary_key, boundary_points, half_extents)
    boundary_xyz = _xy_to_xyz(boundary_xy)

    true_distances, true_gradients = rectangle_signed_distance_and_gradient(
        points,
        width=rectangle.width,
        height=rectangle.height,
    )
    predicted_distances, predicted_gradients_xyz = legacy_sdf_values_and_gradients(params, points)
    predicted_gradients = predicted_gradients_xyz[:, :2]
    boundary_predictions = legacy_sdf_values(params, boundary_xyz)

    true_distances_np = np.asarray(jax.device_get(true_distances), dtype=np.float64)
    predicted_distances_np = np.asarray(jax.device_get(predicted_distances), dtype=np.float64)
    true_gradients_np = np.asarray(jax.device_get(true_gradients), dtype=np.float64)
    predicted_gradients_np = np.asarray(jax.device_get(predicted_gradients), dtype=np.float64)
    predicted_gradients_xyz_np = np.asarray(
        jax.device_get(predicted_gradients_xyz),
        dtype=np.float64,
    )
    boundary_predictions_np = np.asarray(jax.device_get(boundary_predictions), dtype=np.float64)

    errors = predicted_distances_np - true_distances_np
    absolute_errors = np.abs(errors)
    non_boundary_mask = np.abs(true_distances_np) > 1e-6
    collision_mask = true_distances_np < -1e-6
    free_mask = true_distances_np > 1e-6
    near_mask = np.abs(true_distances_np) <= near_band
    false_safe_mask = collision_mask & (predicted_distances_np >= 0.0)
    false_collision_mask = free_mask & (predicted_distances_np <= 0.0)

    predicted_gradient_norms = np.linalg.norm(predicted_gradients_np, axis=1)
    true_gradient_norms = np.linalg.norm(true_gradients_np, axis=1)
    gradient_mask = true_gradient_norms > 0.5
    cosine_denominator = np.maximum(
        predicted_gradient_norms * true_gradient_norms,
        1e-12,
    )
    gradient_cosines = np.sum(
        predicted_gradients_np * true_gradients_np,
        axis=1,
    ) / cosine_denominator
    gradient_errors = np.linalg.norm(predicted_gradients_np - true_gradients_np, axis=1)

    near_surface_max_overestimation = max(0.0, _masked_max(errors, near_mask))
    contact_shell = evaluate_rectangle_contact_shell(
        params,
        rectangle,
        scales=contact_shell_scales,
        radii=contact_shell_radii,
        points_per_shell=contact_shell_points,
        seed=seed + 1,
    )
    recommended_margin = max(
        near_surface_max_overestimation,
        float(contact_shell["max_overestimation"]),
    )

    metrics: dict[str, Any] = {
        "validation_points": int(num_points),
        "boundary_validation_points": int(boundary_points),
        "distance_mae": float(np.mean(absolute_errors)),
        "distance_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "distance_max_abs_error": float(np.max(absolute_errors)),
        "distance_bias": float(np.mean(errors)),
        "near_surface_band": float(near_band),
        "near_surface_points": int(np.count_nonzero(near_mask)),
        "near_surface_mae": _masked_mean(absolute_errors, near_mask),
        "near_surface_max_abs_error": _masked_max(absolute_errors, near_mask),
        "near_surface_max_overestimation": near_surface_max_overestimation,
        "collision_max_overestimation": max(0.0, _masked_max(errors, collision_mask)),
        # This empirical margin covers positive errors observed in both the
        # sampled near-surface set and configured exact-contact circle shells.
        # It is not a formal approximation bound.
        "recommended_conservative_margin": recommended_margin,
        "sign_accuracy": _masked_mean(
            np.sign(predicted_distances_np) == np.sign(true_distances_np),
            non_boundary_mask,
        ),
        "collision_points": int(np.count_nonzero(collision_mask)),
        "false_safe_count": int(np.count_nonzero(false_safe_mask)),
        "false_safe_rate": _masked_mean(false_safe_mask, collision_mask),
        "free_points": int(np.count_nonzero(free_mask)),
        "false_collision_count": int(np.count_nonzero(false_collision_mask)),
        "false_collision_rate": _masked_mean(false_collision_mask, free_mask),
        "boundary_mae": float(np.mean(np.abs(boundary_predictions_np))),
        "boundary_max_abs_error": float(np.max(np.abs(boundary_predictions_np))),
        "boundary_bias": float(np.mean(boundary_predictions_np)),
        "gradient_l2_error_mean": _masked_mean(gradient_errors, gradient_mask),
        "gradient_cosine_mean": _masked_mean(gradient_cosines, gradient_mask),
        "eikonal_abs_error_mean": float(np.mean(np.abs(predicted_gradient_norms - 1.0))),
        "eikonal_abs_error_max": float(np.max(np.abs(predicted_gradient_norms - 1.0))),
        "z_gradient_abs_mean": float(np.mean(np.abs(predicted_gradients_xyz_np[:, 2]))),
        "contact_shell": contact_shell,
    }
    return metrics


def evaluate_rectangle_contact_shell(
    params: Mapping[str, Any],
    rectangle: RectangleSpec,
    *,
    scales: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
    radii: tuple[float, ...] = (0.28, 0.62),
    points_per_shell: int = 2_048,
    seed: int = 20_000,
) -> dict[str, Any]:
    """Evaluate circle centers at exact contact with scaled rectangles.

    A center on the radius-offset shell has exact barrier value zero. For a
    scale ``s`` and circle radius ``r``, the fast-path neural barrier is
    ``s * f(center / s) - r``. This deliberately exercises far-field canonical
    queries produced by small links and large obstacles.
    """
    validate_legacy_params(params)
    if not scales or any(not np.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("scales must contain positive finite values")
    if not radii or any(not np.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError("radii must contain positive finite values")
    if points_per_shell < 8:
        raise ValueError("points_per_shell must be at least 8")

    cases: list[dict[str, float | int]] = []
    max_overestimation = 0.0
    for scale_index, scale in enumerate(scales):
        scaled_half_extents = np.asarray(
            (0.5 * rectangle.width * scale, 0.5 * rectangle.height * scale),
            dtype=np.float32,
        )
        for radius_index, radius in enumerate(radii):
            case_seed = seed + 1_009 * scale_index + 9_173 * radius_index
            centers = _sample_rectangle_offset_shell_numpy(
                case_seed,
                points_per_shell,
                scaled_half_extents,
                radius,
            )
            exact_barriers = np.asarray(
                jax.device_get(
                    rectangle_signed_distance(
                        centers,
                        width=rectangle.width * scale,
                        height=rectangle.height * scale,
                    )
                    - radius
                ),
                dtype=np.float64,
            )
            canonical_queries = jnp.asarray(centers / scale, dtype=NSDF_DTYPE)
            predictions = np.asarray(
                jax.device_get(scale * legacy_sdf_values(params, canonical_queries) - radius),
                dtype=np.float64,
            )
            case_max_overestimation = max(0.0, float(np.max(predictions)))
            max_overestimation = max(max_overestimation, case_max_overestimation)
            cases.append(
                {
                    "scale": float(scale),
                    "radius": float(radius),
                    "points": int(points_per_shell),
                    "canonical_exact_distance": float(radius / scale),
                    "exact_contact_max_abs_error": float(np.max(np.abs(exact_barriers))),
                    "barrier_mae": float(np.mean(np.abs(predictions))),
                    "barrier_bias": float(np.mean(predictions)),
                    "max_overestimation": case_max_overestimation,
                    "max_underestimation": max(0.0, float(-np.min(predictions))),
                    "positive_fraction_at_contact": float(np.mean(predictions > 0.0)),
                }
            )
    return {
        "description": "scaled_network(center / scale) - circle_radius at exact contact",
        "max_overestimation": max_overestimation,
        "cases": cases,
    }


def save_training_artifacts(
    result: TrainingResult,
    checkpoint_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    source_revision: str | None = None,
    source_provenance: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save a new legacy-compatible checkpoint and JSON metadata without overwrite."""
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix != ".npy":
        raise ValueError(f"checkpoint path must end in .npy: {checkpoint}")
    metadata = Path(metadata_path) if metadata_path is not None else checkpoint.with_suffix(".json")
    if metadata.suffix != ".json":
        raise ValueError(f"metadata path must end in .json: {metadata}")
    if checkpoint.exists() or metadata.exists():
        existing = [str(path) for path in (checkpoint, metadata) if path.exists()]
        raise FileExistsError(f"refusing to overwrite existing training artifact(s): {existing}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    validate_legacy_params(
        result.params,
        expected_hidden_dim=result.config.hidden_dim,
    )
    numpy_params = params_to_numpy(result.params)
    created_paths: list[Path] = []
    try:
        with checkpoint.open("xb") as checkpoint_file:
            created_paths.append(checkpoint)
            np.save(checkpoint_file, numpy_params, allow_pickle=True)
        checkpoint_sha256 = _sha256(checkpoint)
        metadata_document: dict[str, Any] = {
            "format_version": 1,
            "architecture": {
                "name": f"legacy_sdfnet_4x{result.config.hidden_dim}",
                "input_dim": 3,
                "hidden_dim": result.config.hidden_dim,
                "output_dim": 1,
                "layer_order": list(LEGACY_ARCHITECTURE_ORDER),
                "dtype": "float32",
            },
            "geometry": {
                "type": "axis_aligned_rectangle",
                "name": result.rectangle.name,
                "width": result.rectangle.width,
                "height": result.rectangle.height,
                "units": result.rectangle.units,
            },
            "training": {
                "objective": "distance_mse + eikonal_weight * planar_xy_eikonal",
                "sampling": "equal uniform_balanced_sign / near_surface / boundary",
                "config": asdict(result.config),
                "loss_history": list(result.loss_history),
            },
            "validation": result.validation_metrics,
            "margin_scope": {
                "type": "empirical_sampled",
                "covers": [
                    "validation.near_surface_max_overestimation",
                    "validation.contact_shell.max_overestimation",
                ],
                "excludes": [
                    "validation.collision_max_overestimation",
                    "formal_global_error_bound",
                ],
                "description": (
                    "recommended_conservative_margin is the sampled maximum "
                    "positive error over near-surface and configured exact-contact "
                    "shell queries only"
                ),
            },
            "checkpoint": {
                "file": checkpoint.name,
                "sha256": checkpoint_sha256,
            },
            "provenance": {
                "source_revision": source_revision,
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
            },
        }
        if source_provenance is not None:
            metadata_document["provenance"]["source"] = _json_compatible(
                source_provenance
            )
        if extra_metadata:
            metadata_document["comparison"] = _json_compatible(extra_metadata)
        with metadata.open("x", encoding="utf-8") as metadata_file:
            created_paths.append(metadata)
            json.dump(metadata_document, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
    except BaseException:
        for created_path in reversed(created_paths):
            if created_path.exists():
                created_path.unlink()
        raise
    return checkpoint, metadata


def _sample_balanced_points(
    key: jax.Array,
    count: int,
    half_extents: jax.Array,
    domain_half_extents: jax.Array,
    near_band: float,
) -> jax.Array:
    uniform_count, near_count, boundary_count = _split_three(count)
    uniform_key, near_key, boundary_key, shuffle_key = jax.random.split(key, 4)
    uniform_xy = _sample_uniform_inside_outside(
        uniform_key,
        uniform_count,
        half_extents,
        domain_half_extents,
    )
    near_xy = _sample_near_surface(near_key, near_count, half_extents, near_band)
    boundary_xy, _ = _sample_rectangle_boundary(boundary_key, boundary_count, half_extents)
    xy = jnp.concatenate((uniform_xy, near_xy, boundary_xy), axis=0)
    permutation = jax.random.permutation(shuffle_key, count)
    return _xy_to_xyz(xy[permutation])


def _sample_eikonal_points(
    key: jax.Array,
    count: int,
    half_extents: jax.Array,
    domain_half_extents: jax.Array,
    near_band: float,
) -> jax.Array:
    uniform_count = count // 2
    near_count = count - uniform_count
    uniform_key, near_key, shuffle_key = jax.random.split(key, 3)
    uniform_xy = _sample_uniform_inside_outside(
        uniform_key,
        uniform_count,
        half_extents,
        domain_half_extents,
    )
    near_xy = _sample_near_surface(near_key, near_count, half_extents, near_band)
    xy = jnp.concatenate((uniform_xy, near_xy), axis=0)
    return _xy_to_xyz(xy[jax.random.permutation(shuffle_key, count)])


def _sample_uniform_inside_outside(
    key: jax.Array,
    count: int,
    half_extents: jax.Array,
    domain_half_extents: jax.Array,
) -> jax.Array:
    inside_count = count // 2
    outside_count = count - inside_count
    inside_key, outside_key = jax.random.split(key)
    inside = jax.random.uniform(
        inside_key,
        (inside_count, 2),
        minval=-half_extents,
        maxval=half_extents,
        dtype=NSDF_DTYPE,
    )
    outside = _sample_uniform_outside_rectangle(
        outside_key,
        outside_count,
        half_extents,
        domain_half_extents,
    )
    return jnp.concatenate((inside, outside), axis=0)


def _sample_uniform_outside_rectangle(
    key: jax.Array,
    count: int,
    half_extents: jax.Array,
    domain_half_extents: jax.Array,
) -> jax.Array:
    region_key, coordinate_key = jax.random.split(key)
    hx, hy = half_extents
    dx, dy = domain_half_extents
    margin_x = dx - hx
    margin_y = dy - hy
    region_areas = jnp.asarray(
        (
            margin_x * (2.0 * dy),
            margin_x * (2.0 * dy),
            (2.0 * hx) * margin_y,
            (2.0 * hx) * margin_y,
        ),
        dtype=NSDF_DTYPE,
    )
    cumulative_probability = jnp.cumsum(region_areas) / jnp.sum(region_areas)
    region_uniform = jax.random.uniform(region_key, (count,), dtype=NSDF_DTYPE)
    region = jnp.sum(
        region_uniform[:, None] > cumulative_probability[:-1][None, :],
        axis=1,
    )
    coordinates = jax.random.uniform(coordinate_key, (count, 2), dtype=NSDF_DTYPE)
    u, v = coordinates[:, 0], coordinates[:, 1]
    candidates = jnp.stack(
        (
            jnp.column_stack((-dx + u * margin_x, -dy + v * (2.0 * dy))),
            jnp.column_stack((hx + u * margin_x, -dy + v * (2.0 * dy))),
            jnp.column_stack((-hx + u * (2.0 * hx), -dy + v * margin_y)),
            jnp.column_stack((-hx + u * (2.0 * hx), hy + v * margin_y)),
        ),
        axis=1,
    )
    return candidates[jnp.arange(count), region]


def _sample_rectangle_boundary(
    key: jax.Array,
    count: int,
    half_extents: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    hx, hy = half_extents
    perimeter = 4.0 * (hx + hy)
    coordinate = jax.random.uniform(
        key,
        (count,),
        minval=0.0,
        maxval=perimeter,
        dtype=NSDF_DTYPE,
    )
    bottom_end = 2.0 * hx
    right_end = bottom_end + 2.0 * hy
    top_end = right_end + 2.0 * hx

    bottom = coordinate < bottom_end
    right = (coordinate >= bottom_end) & (coordinate < right_end)
    top = (coordinate >= right_end) & (coordinate < top_end)

    bottom_points = jnp.column_stack((-hx + coordinate, jnp.full_like(coordinate, -hy)))
    right_offset = coordinate - bottom_end
    right_points = jnp.column_stack((jnp.full_like(coordinate, hx), -hy + right_offset))
    top_offset = coordinate - right_end
    top_points = jnp.column_stack((hx - top_offset, jnp.full_like(coordinate, hy)))
    left_offset = coordinate - top_end
    left_points = jnp.column_stack((jnp.full_like(coordinate, -hx), hy - left_offset))

    points = jnp.where(
        bottom[:, None],
        bottom_points,
        jnp.where(right[:, None], right_points, jnp.where(top[:, None], top_points, left_points)),
    )
    normals = jnp.where(
        bottom[:, None],
        jnp.asarray((0.0, -1.0), dtype=NSDF_DTYPE),
        jnp.where(
            right[:, None],
            jnp.asarray((1.0, 0.0), dtype=NSDF_DTYPE),
            jnp.where(
                top[:, None],
                jnp.asarray((0.0, 1.0), dtype=NSDF_DTYPE),
                jnp.asarray((-1.0, 0.0), dtype=NSDF_DTYPE),
            ),
        ),
    )
    return points, normals


def _sample_near_surface(
    key: jax.Array,
    count: int,
    half_extents: jax.Array,
    near_band: float,
) -> jax.Array:
    boundary_key, magnitude_key, sign_key = jax.random.split(key, 3)
    boundary_points, normals = _sample_rectangle_boundary(boundary_key, count, half_extents)
    magnitudes = jax.random.uniform(
        magnitude_key,
        (count,),
        minval=0.0,
        maxval=near_band,
        dtype=NSDF_DTYPE,
    )
    signs = jnp.where(
        jax.random.bernoulli(sign_key, 0.5, (count,)),
        1.0,
        -1.0,
    ).astype(NSDF_DTYPE)
    return boundary_points + (signs * magnitudes)[:, None] * normals


def _sample_rectangle_offset_shell_numpy(
    seed: int,
    count: int,
    half_extents: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Sample uniformly by arc length from a rounded rectangle boundary."""
    hx, hy = (float(value) for value in half_extents)
    straight_x = 2.0 * hx
    straight_y = 2.0 * hy
    quarter_arc = 0.5 * np.pi * radius
    segment_lengths = np.asarray(
        (
            straight_x,
            quarter_arc,
            straight_y,
            quarter_arc,
            straight_x,
            quarter_arc,
            straight_y,
            quarter_arc,
        ),
        dtype=np.float64,
    )
    cumulative = np.cumsum(segment_lengths)
    rng = np.random.default_rng(seed)
    coordinate = rng.uniform(0.0, cumulative[-1], size=count)
    segment = np.searchsorted(cumulative, coordinate, side="right")
    starts = np.concatenate((np.zeros((1,), dtype=np.float64), cumulative[:-1]))
    offset = coordinate - starts[segment]

    points = np.empty((count, 2), dtype=np.float64)
    for segment_index in range(8):
        selected = segment == segment_index
        local = offset[selected]
        if segment_index == 0:
            points[selected] = np.column_stack((-hx + local, np.full_like(local, -hy - radius)))
        elif segment_index == 1:
            angle = -0.5 * np.pi + local / radius
            points[selected] = np.column_stack(
                (hx + radius * np.cos(angle), -hy + radius * np.sin(angle))
            )
        elif segment_index == 2:
            points[selected] = np.column_stack(
                (np.full_like(local, hx + radius), -hy + local)
            )
        elif segment_index == 3:
            angle = local / radius
            points[selected] = np.column_stack(
                (hx + radius * np.cos(angle), hy + radius * np.sin(angle))
            )
        elif segment_index == 4:
            points[selected] = np.column_stack((hx - local, np.full_like(local, hy + radius)))
        elif segment_index == 5:
            angle = 0.5 * np.pi + local / radius
            points[selected] = np.column_stack(
                (-hx + radius * np.cos(angle), hy + radius * np.sin(angle))
            )
        elif segment_index == 6:
            points[selected] = np.column_stack(
                (np.full_like(local, -hx - radius), hy - local)
            )
        else:
            angle = np.pi + local / radius
            points[selected] = np.column_stack(
                (-hx + radius * np.cos(angle), -hy + radius * np.sin(angle))
            )
    return points.astype(np.float32)


def _adam_update(
    params: Mapping[str, Any],
    gradients: Mapping[str, Any],
    first_moment: Mapping[str, Any],
    second_moment: Mapping[str, Any],
    step: jax.Array,
    learning_rate: jax.Array,
    config: TrainingConfig,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    one = jnp.asarray(1.0, dtype=NSDF_DTYPE)
    beta1 = jnp.asarray(config.adam_beta1, dtype=NSDF_DTYPE)
    beta2 = jnp.asarray(config.adam_beta2, dtype=NSDF_DTYPE)
    epsilon = jnp.asarray(config.adam_epsilon, dtype=NSDF_DTYPE)
    learning_rate = jnp.asarray(learning_rate, dtype=NSDF_DTYPE)
    next_first_moment = jax.tree.map(
        lambda moment, gradient: beta1 * moment + (one - beta1) * gradient,
        first_moment,
        gradients,
    )
    next_second_moment = jax.tree.map(
        lambda moment, gradient: beta2 * moment
        + (one - beta2) * jnp.square(gradient),
        second_moment,
        gradients,
    )
    first_correction = one - jnp.power(beta1, step)
    second_correction = one - jnp.power(beta2, step)
    next_params = jax.tree.map(
        lambda parameter, moment, squared_moment: parameter
        - learning_rate
        * (moment / first_correction)
        / (jnp.sqrt(squared_moment / second_correction) + epsilon),
        params,
        next_first_moment,
        next_second_moment,
    )
    return next_params, next_first_moment, next_second_moment


def _clip_gradients(
    gradients: Mapping[str, Any],
    max_norm: float,
) -> tuple[Mapping[str, Any], jax.Array]:
    one = jnp.asarray(1.0, dtype=NSDF_DTYPE)
    minimum_norm = jnp.asarray(1e-12, dtype=NSDF_DTYPE)
    max_norm_array = jnp.asarray(max_norm, dtype=NSDF_DTYPE)
    squared_norm = sum(
        jnp.sum(jnp.square(gradient))
        for gradient in jax.tree.leaves(gradients)
    )
    norm = jnp.sqrt(squared_norm)
    scale = jnp.minimum(one, max_norm_array / jnp.maximum(norm, minimum_norm))
    return jax.tree.map(lambda gradient: gradient * scale, gradients), norm


def _cosine_learning_rate(step: jax.Array, config: TrainingConfig) -> jax.Array:
    one = jnp.asarray(1.0, dtype=NSDF_DTYPE)
    half = jnp.asarray(0.5, dtype=NSDF_DTYPE)
    minimum_ratio = jnp.asarray(config.min_learning_rate_ratio, dtype=NSDF_DTYPE)
    progress = jnp.minimum(
        jnp.asarray(step, dtype=NSDF_DTYPE)
        / jnp.asarray(config.steps, dtype=NSDF_DTYPE),
        one,
    )
    cosine = half * (one + jnp.cos(jnp.asarray(jnp.pi, dtype=NSDF_DTYPE) * progress))
    ratio = minimum_ratio + (one - minimum_ratio) * cosine
    return jnp.asarray(config.learning_rate, dtype=NSDF_DTYPE) * ratio


def _resolved_near_band(rectangle: RectangleSpec, requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    return 0.1 * min(rectangle.width, rectangle.height)


def _points_to_xy(points: jax.Array) -> jax.Array:
    array = jnp.atleast_2d(jnp.asarray(points, dtype=NSDF_DTYPE))
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(f"points must have shape (N, 2) or (N, 3), got {array.shape}")
    return array[:, :2]


def _xy_to_xyz(points: jax.Array) -> jax.Array:
    return jnp.column_stack(
        (
            points,
            jnp.zeros((points.shape[0],), dtype=NSDF_DTYPE),
        )
    )


def _split_three(count: int) -> tuple[int, int, int]:
    first = count // 3
    second = count // 3
    return first, second, count - first - second


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(np.mean(selected)) if selected.size else float("nan")


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(np.max(selected)) if selected.size else float("nan")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
