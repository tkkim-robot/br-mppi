from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


NSDF_DTYPE = jnp.float32
LEGACY_LAYER_NAMES = ("Dense_0", "Dense_1", "Dense_2", "Dense_3")
LEGACY_ARCHITECTURE_ORDER = (
    "Dense_0",
    "Dense_1",
    "softplus",
    "Dense_2",
    "softplus",
    "Dense_3",
)


def init_legacy_sdf_params(
    key: jax.Array,
    *,
    hidden_dim: int = 16,
) -> dict[str, dict[str, dict[str, jax.Array]]]:
    """Initialize the legacy four-dense-layer SDF network in float32.

    The historical checkpoint has an intentionally unusual activation order:
    ``Dense_0 -> Dense_1 -> softplus -> Dense_2 -> softplus -> Dense_3``.
    Keeping that order makes newly trained ``.npy`` files directly compatible
    with :class:`sdf.pretrained_sdf.PretrainedShapeSDF`.
    """
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")

    dimensions = (
        (3, hidden_dim),
        (hidden_dim, hidden_dim),
        (hidden_dim, hidden_dim),
        (hidden_dim, 1),
    )
    layer_keys = jax.random.split(key, len(dimensions))
    layers: dict[str, dict[str, jax.Array]] = {}
    for name, layer_key, (fan_in, fan_out) in zip(
        LEGACY_LAYER_NAMES,
        layer_keys,
        dimensions,
        strict=True,
    ):
        limit = jnp.sqrt(jnp.asarray(6.0 / (fan_in + fan_out), dtype=NSDF_DTYPE))
        kernel = jax.random.uniform(
            layer_key,
            (fan_in, fan_out),
            minval=-limit,
            maxval=limit,
            dtype=NSDF_DTYPE,
        )
        layers[name] = {
            "kernel": kernel,
            "bias": jnp.zeros((fan_out,), dtype=NSDF_DTYPE),
        }
    return {"params": layers}


def legacy_sdf_values(params: Mapping[str, Any], points: jax.Array) -> jax.Array:
    """Evaluate legacy SDFNet parameters at 2-D or 3-D local-frame points."""
    local_points = _points_to_3d(points)
    weights = params["params"]
    w0, b0 = _dense_params(weights, "Dense_0")
    w1, b1 = _dense_params(weights, "Dense_1")
    w2, b2 = _dense_params(weights, "Dense_2")
    w3, b3 = _dense_params(weights, "Dense_3")

    z0 = local_points @ w0 + b0
    z1 = z0 @ w1 + b1
    a1 = _softplus(z1)
    z2 = a1 @ w2 + b2
    a2 = _softplus(z2)
    return (a2 @ w3 + b3).reshape(-1)


def legacy_sdf_values_and_gradients(
    params: Mapping[str, Any],
    points: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return SDF values and exact gradients with respect to local XYZ."""
    local_points = _points_to_3d(points)
    weights = params["params"]
    w0, b0 = _dense_params(weights, "Dense_0")
    w1, b1 = _dense_params(weights, "Dense_1")
    w2, b2 = _dense_params(weights, "Dense_2")
    w3, b3 = _dense_params(weights, "Dense_3")

    z0 = local_points @ w0 + b0
    z1 = z0 @ w1 + b1
    a1 = _softplus(z1)
    z2 = a1 @ w2 + b2
    a2 = _softplus(z2)
    values = (a2 @ w3 + b3).reshape(-1)

    grad_z2 = _sigmoid(z2) * w3.reshape(1, -1)
    grad_a1 = grad_z2 @ w2.T
    grad_z1 = grad_a1 * _sigmoid(z1)
    grad_z0 = grad_z1 @ w1.T
    gradients = grad_z0 @ w0.T
    return values, gradients


def params_to_jax(params: Mapping[str, Any]) -> dict[str, dict[str, dict[str, jax.Array]]]:
    """Convert a validated legacy checkpoint tree to float32 JAX arrays."""
    hidden_dim = validate_legacy_params(params)
    del hidden_dim
    converted: dict[str, dict[str, dict[str, jax.Array]]] = {"params": {}}
    for layer_name in LEGACY_LAYER_NAMES:
        layer = params["params"][layer_name]
        converted["params"][layer_name] = {
            "kernel": jnp.asarray(layer["kernel"], dtype=NSDF_DTYPE),
            "bias": jnp.asarray(layer["bias"], dtype=NSDF_DTYPE),
        }
    return converted


def params_to_numpy(params: Mapping[str, Any]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Convert a legacy parameter tree to portable float32 NumPy arrays."""
    hidden_dim = validate_legacy_params(params)
    del hidden_dim
    converted: dict[str, dict[str, dict[str, np.ndarray]]] = {"params": {}}
    for layer_name in LEGACY_LAYER_NAMES:
        layer = params["params"][layer_name]
        converted["params"][layer_name] = {
            "kernel": np.asarray(jax.device_get(layer["kernel"]), dtype=np.float32),
            "bias": np.asarray(jax.device_get(layer["bias"]), dtype=np.float32),
        }
    return converted


def validate_legacy_params(
    params: Mapping[str, Any],
    *,
    expected_hidden_dim: int | None = None,
) -> int:
    """Validate exact float32 legacy parameter schema and return hidden width."""
    if not isinstance(params, Mapping) or not isinstance(params.get("params"), Mapping):
        raise ValueError("checkpoint must contain a mapping at key 'params'")
    layers = params["params"]
    missing = [name for name in LEGACY_LAYER_NAMES if name not in layers]
    extra = [name for name in layers if name not in LEGACY_LAYER_NAMES]
    if missing or extra:
        raise ValueError(f"checkpoint layers mismatch: missing={missing}, extra={extra}")

    first_layer = layers["Dense_0"]
    if (
        not isinstance(first_layer, Mapping)
        or set(first_layer) != {"kernel", "bias"}
    ):
        raise ValueError("Dense_0 must contain exactly 'kernel' and 'bias'")
    first_kernel = np.shape(first_layer["kernel"])
    if len(first_kernel) != 2 or first_kernel[0] != 3 or first_kernel[1] <= 0:
        raise ValueError(f"Dense_0 kernel must have shape (3, hidden_dim), got {first_kernel}")
    hidden_dim = int(first_kernel[1])
    if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
        raise ValueError(
            f"checkpoint hidden_dim must be {expected_hidden_dim}, got {hidden_dim}"
        )
    expected = {
        "Dense_0": ((3, hidden_dim), (hidden_dim,)),
        "Dense_1": ((hidden_dim, hidden_dim), (hidden_dim,)),
        "Dense_2": ((hidden_dim, hidden_dim), (hidden_dim,)),
        "Dense_3": ((hidden_dim, 1), (1,)),
    }
    for name, (kernel_shape, bias_shape) in expected.items():
        layer = layers[name]
        if not isinstance(layer, Mapping) or set(layer) != {"kernel", "bias"}:
            raise ValueError(f"{name} must contain exactly 'kernel' and 'bias'")
        actual_kernel_shape = np.shape(layer["kernel"])
        actual_bias_shape = np.shape(layer["bias"])
        if actual_kernel_shape != kernel_shape or actual_bias_shape != bias_shape:
            raise ValueError(
                f"{name} shapes must be kernel={kernel_shape}, bias={bias_shape}; "
                f"got kernel={actual_kernel_shape}, bias={actual_bias_shape}"
            )
        for parameter_name in ("kernel", "bias"):
            parameter = layer[parameter_name]
            values = np.asarray(jax.device_get(parameter))
            dtype = values.dtype
            if dtype != np.dtype(np.float32):
                raise ValueError(
                    f"{name}.{parameter_name} must use float32, got {dtype}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name}.{parameter_name} must contain only finite values"
                )
    return hidden_dim


def load_legacy_checkpoint(path: str | Path) -> dict[str, dict[str, dict[str, jax.Array]]]:
    """Load a trusted legacy ``.npy`` parameter mapping as float32 JAX arrays."""
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".npy":
        raise ValueError(f"expected a .npy checkpoint, got: {checkpoint_path}")
    raw = np.load(checkpoint_path, allow_pickle=True).item()
    return params_to_jax(raw)


def _dense_params(
    params: Mapping[str, Any],
    name: str,
) -> tuple[jax.Array, jax.Array]:
    layer = params[name]
    return layer["kernel"], layer["bias"]


def _points_to_3d(points: jax.Array) -> jax.Array:
    local_points = jnp.atleast_2d(jnp.asarray(points, dtype=NSDF_DTYPE))
    if local_points.ndim != 2 or local_points.shape[1] not in (2, 3):
        raise ValueError(f"points must have shape (N, 2) or (N, 3), got {local_points.shape}")
    if local_points.shape[1] == 2:
        local_points = jnp.column_stack(
            (local_points, jnp.zeros((local_points.shape[0],), dtype=NSDF_DTYPE))
        )
    return local_points


def _softplus(values: jax.Array) -> jax.Array:
    return jnp.log1p(jnp.exp(-jnp.abs(values))) + jnp.maximum(values, 0.0)


def _sigmoid(values: jax.Array) -> jax.Array:
    positive = values >= 0.0
    exp_neg_abs = jnp.exp(-jnp.abs(values))
    return jnp.where(
        positive,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )
