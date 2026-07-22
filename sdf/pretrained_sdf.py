from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sdf.geometry import ObstacleField


NSDF_DTYPE = jnp.float32


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
        self._field_cache_key: tuple[int, tuple[tuple[float, float, float], ...]] | None = None
        self._field_cache_points: np.ndarray | None = None

    @property
    def description(self) -> str:
        return (
            f"{self.spec.model_file} ({self.spec.shape_name}, "
            f"source shape index {self.spec.source_shape_index})"
        )

    def signed_distance(self, points: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the checkpoint in its local shape frame."""
        local_points = jnp.atleast_2d(jnp.asarray(points, dtype=NSDF_DTYPE))
        local_points = _pad_points_to_3d(local_points)
        return self._values(local_points)

    def obstacle_barriers(self, state: jnp.ndarray, field: ObstacleField) -> jnp.ndarray:
        """Return one robot-shape SDF barrier value per obstacle."""
        obstacle_points = self._obstacle_points(field)
        state = jnp.asarray(state, dtype=NSDF_DTYPE)
        x, y = state[:2]
        theta = state[2] if state.shape[0] >= 3 else jnp.array(0.0, dtype=NSDF_DTYPE)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=NSDF_DTYPE)
        local_xy = (obstacle_points - jnp.array([x, y], dtype=NSDF_DTYPE)) @ rot
        local_points = jnp.column_stack((local_xy, jnp.zeros(local_xy.shape[0], dtype=NSDF_DTYPE)))
        values = self._values(local_points)
        values = values.reshape((len(field.obstacles), self.points_per_obstacle))
        return jnp.min(values, axis=1)

    def obstacle_barriers_and_jacobian(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return per-obstacle barriers and their exact robot-state Jacobian.

        The network and its local gradients are evaluated once. The selected
        surface point for each obstacle is then differentiated through the
        world-to-robot rigid transform. State components after ``(x, y, yaw)``
        do not directly affect the shape SDF and therefore have zero columns.
        """
        obstacle_points = self._obstacle_points(field)
        state = jnp.asarray(state, dtype=NSDF_DTYPE)
        x, y = state[:2]
        theta = state[2] if state.shape[0] >= 3 else jnp.array(0.0, dtype=NSDF_DTYPE)
        c, s = jnp.cos(theta), jnp.sin(theta)
        rot = jnp.array([[c, -s], [s, c]], dtype=NSDF_DTYPE)
        local_xy = (obstacle_points - jnp.array([x, y], dtype=NSDF_DTYPE)) @ rot
        local_points = jnp.column_stack((local_xy, jnp.zeros(local_xy.shape[0], dtype=NSDF_DTYPE)))
        point_values, point_local_gradients = self._value_and_local_gradient(local_points)

        obstacle_count = len(field.obstacles)
        values = point_values.reshape((obstacle_count, self.points_per_obstacle))
        local_gradients = point_local_gradients[:, :2].reshape(
            (obstacle_count, self.points_per_obstacle, 2)
        )
        local_coordinates = local_xy.reshape((obstacle_count, self.points_per_obstacle, 2))
        nearest = jnp.argmin(values, axis=1)
        row = jnp.arange(obstacle_count)
        chosen_values = values[row, nearest]
        chosen_gradients = local_gradients[row, nearest]
        chosen_coordinates = local_coordinates[row, nearest]

        gx, gy = chosen_gradients[:, 0], chosen_gradients[:, 1]
        local_x, local_y = chosen_coordinates[:, 0], chosen_coordinates[:, 1]
        jacobian = jnp.zeros((obstacle_count, state.shape[0]), dtype=NSDF_DTYPE)
        jacobian = jacobian.at[:, 0].set(-gx * c + gy * s)
        jacobian = jacobian.at[:, 1].set(-gx * s - gy * c)
        if state.shape[0] >= 3:
            jacobian = jacobian.at[:, 2].set(gx * local_y - gy * local_x)
        return chosen_values, jacobian

    def _obstacle_points(self, field: ObstacleField) -> jnp.ndarray:
        obstacle_key = tuple(
            (float(center[0]), float(center[1]), obstacle.radius)
            for obstacle in field.obstacles
            for center in (np.asarray(obstacle.center, dtype=np.float64),)
        )
        cache_key = (self.points_per_obstacle, obstacle_key)
        if cache_key != self._field_cache_key:
            theta = np.linspace(
                0.0,
                2.0 * np.pi,
                self.points_per_obstacle,
                endpoint=False,
            )
            unit_circle = np.stack((np.cos(theta), np.sin(theta)), axis=1)
            centers = np.asarray([[x, y] for x, y, _radius in obstacle_key], dtype=np.float64)
            radii = np.asarray([radius for _x, _y, radius in obstacle_key], dtype=np.float64)
            self._field_cache_key = cache_key
            self._field_cache_points = (
                centers[:, None, :] + radii[:, None, None] * unit_circle[None, :, :]
            ).reshape((-1, 2)).astype(np.float32)
        assert self._field_cache_points is not None
        return jnp.asarray(self._field_cache_points, dtype=NSDF_DTYPE)

    def _values(self, points: jnp.ndarray) -> jnp.ndarray:
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
        return (a2 @ w3 + b3).reshape(-1)

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
            "kernel": jnp.asarray(layer["kernel"], dtype=NSDF_DTYPE),
            "bias": jnp.asarray(layer["bias"], dtype=NSDF_DTYPE),
        }
    return converted


def _dense_params(params: dict, name: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    layer = params[name]
    return layer["kernel"], layer["bias"]


def _pad_points_to_3d(points: jnp.ndarray) -> jnp.ndarray:
    if points.shape[1] == 2:
        return jnp.column_stack((points, jnp.zeros(points.shape[0], dtype=NSDF_DTYPE)))
    return points


def _softplus(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    positive = x >= 0.0
    exp_neg_abs = jnp.exp(-jnp.abs(x))
    return jnp.where(positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))
