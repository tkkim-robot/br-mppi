from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sdf.geometry import ObstacleField


class PretrainedSDFUnavailable(RuntimeError):
    """Raised when no matching pretrained SDF asset is available."""


@dataclass(frozen=True)
class PretrainedSDFSpec:
    model_file: str
    shape_name: str
    source_shape_index: int


SUPPORTED_PRETRAINED_SDFS = {
    "unicycle": PretrainedSDFSpec("link1_model_4_16.npy", "rectangle 1.0 x 0.4", 0),
    "dynamic_unicycle": PretrainedSDFSpec("link1_model_4_16.npy", "rectangle 1.0 x 0.4", 0),
    "planar_quadrotor": PretrainedSDFSpec("link7_model_4_16.npy", "quad2 0.56 x 0.28", 6),
}

UNSUPPORTED_PRETRAINED_SDFS = {
    "single_integrator": "no pretrained disk/point SDF was found in sdf/trained_models",
    "mobile_arm": (
        "the available pretrained files are canonical single-shape SDFs, not a model for "
        "the original mobile-arm base plus two fixed 4-link arms"
    ),
}


@dataclass
class PretrainedShapeSDF:
    """JAX evaluator for the vendored Flax SDFNet checkpoints."""

    robot_name: str
    spec: PretrainedSDFSpec
    params: dict
    points_per_obstacle: int = 16

    def __post_init__(self) -> None:
        self.params = _params_to_jax(self.params)
        self._field_cache_key: tuple[tuple[float, float, float], ...] | None = None
        self._field_cache_points: jnp.ndarray | None = None

    @property
    def description(self) -> str:
        return (
            f"{self.spec.model_file} ({self.spec.shape_name}, "
            f"source shape index {self.spec.source_shape_index})"
        )

    def signed_distance(self, points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the checkpoint in its local shape frame."""
        local_points = jnp.atleast_2d(jnp.asarray(points, dtype=float))
        local_points = _pad_points_to_3d(local_points)
        return self._value_and_local_gradient(local_points)[0]

    def distance_and_gradient(self, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        local_points = jnp.atleast_2d(jnp.asarray(points, dtype=float))
        local_points = _pad_points_to_3d(local_points)
        values, gradients = self._value_and_local_gradient(local_points)
        return values, gradients[:, :2]

    def clearance_and_gradient(self, state: jnp.ndarray, field: ObstacleField) -> tuple[jnp.ndarray, jnp.ndarray]:
        obstacle_points = self._obstacle_points(field)
        state = jnp.asarray(state, dtype=float)
        x, y = state[:2]
        theta = state[2] if state.shape[0] >= 3 else jnp.array(0.0, dtype=float)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=float)
        local_xy = (obstacle_points - jnp.array([x, y], dtype=float)) @ rot
        local_points = jnp.column_stack((local_xy, jnp.zeros(local_xy.shape[0], dtype=float)))
        values, local_gradients = self._value_and_local_gradient(local_points)
        nearest = jnp.argmin(values)
        value = values[nearest]

        gx, gy = local_gradients[nearest, 0], local_gradients[nearest, 1]
        world_gradient = jnp.array([-gx * c + gy * s, -gx * s - gy * c], dtype=float)
        norm = jnp.linalg.norm(world_gradient)
        world_gradient = jnp.where(norm > 1e-8, world_gradient / jnp.maximum(norm, 1e-8), world_gradient)
        return value, world_gradient

    def obstacle_barriers(self, state: jnp.ndarray, field: ObstacleField) -> jnp.ndarray:
        """Return one robot-shape SDF barrier value per obstacle."""
        obstacle_points = self._obstacle_points(field)
        state = jnp.asarray(state, dtype=float)
        x, y = state[:2]
        theta = state[2] if state.shape[0] >= 3 else jnp.array(0.0, dtype=float)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=float)
        local_xy = (obstacle_points - jnp.array([x, y], dtype=float)) @ rot
        local_points = jnp.column_stack((local_xy, jnp.zeros(local_xy.shape[0], dtype=float)))
        values = self._value_and_local_gradient(local_points)[0]
        values = values.reshape((len(field.obstacles), self.points_per_obstacle))
        return jnp.min(values, axis=1)

    def _obstacle_points(self, field: ObstacleField) -> jnp.ndarray:
        return field.surface_points(self.points_per_obstacle)

    def _value_and_local_gradient(self, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        weights = self.params["params"]
        w0, b0 = _dense_params(weights, "Dense_0")
        w1, b1 = _dense_params(weights, "Dense_1")
        w2, b2 = _dense_params(weights, "Dense_2")
        w3, b3 = _dense_params(weights, "Dense_3")

        z0 = points @ w0 + b0
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


def load_pretrained_sdf_for_robot(robot_name: str, *, repo_root: Path | None = None) -> PretrainedShapeSDF:
    if robot_name in UNSUPPORTED_PRETRAINED_SDFS:
        raise PretrainedSDFUnavailable(UNSUPPORTED_PRETRAINED_SDFS[robot_name])
    if robot_name not in SUPPORTED_PRETRAINED_SDFS:
        raise PretrainedSDFUnavailable(f"no pretrained SDF mapping is configured for robot '{robot_name}'")

    root = repo_root or Path(__file__).resolve().parents[1]
    spec = SUPPORTED_PRETRAINED_SDFS[robot_name]
    model_path = root / "sdf" / "trained_models" / spec.model_file
    if not model_path.exists():
        raise PretrainedSDFUnavailable(f"missing pretrained model file: {model_path}")
    params = np.load(model_path, allow_pickle=True).item()
    return PretrainedShapeSDF(robot_name=robot_name, spec=spec, params=params)


def _params_to_jax(params: dict) -> dict:
    converted = {"params": {}}
    for layer_name, layer in params["params"].items():
        converted["params"][layer_name] = {
            "kernel": jnp.asarray(layer["kernel"], dtype=float),
            "bias": jnp.asarray(layer["bias"], dtype=float),
        }
    return converted


def _dense_params(params: dict, name: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    layer = params[name]
    return layer["kernel"], layer["bias"]


def _pad_points_to_3d(points: jnp.ndarray) -> jnp.ndarray:
    if points.shape[1] == 2:
        return jnp.column_stack((points, jnp.zeros(points.shape[0], dtype=float)))
    return points


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    positive = x >= 0.0
    exp_x = jnp.exp(x)
    return jnp.where(positive, 1.0 / (1.0 + jnp.exp(-x)), exp_x / (1.0 + exp_x))
