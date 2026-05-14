from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    "single_integrator": (
        "no pretrained disk/point SDF was found in sdf/trained_models"
    ),
    "mobile_arm": (
        "the available pretrained files are canonical single-shape SDFs, not a model for "
        "the original mobile-arm base plus two fixed 4-link arms"
    ),
}


@dataclass
class PretrainedShapeSDF:
    """Numpy evaluator for the vendored Flax SDFNet checkpoints."""

    robot_name: str
    spec: PretrainedSDFSpec
    params: dict
    points_per_obstacle: int = 16

    def __post_init__(self) -> None:
        self._field_cache_key: tuple[tuple[float, float, float], ...] | None = None
        self._field_cache_points: np.ndarray | None = None

    @property
    def description(self) -> str:
        return (
            f"{self.spec.model_file} ({self.spec.shape_name}, "
            f"source shape index {self.spec.source_shape_index})"
        )

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the checkpoint in its local shape frame."""
        local_points = np.atleast_2d(points).astype(float)
        if local_points.shape[1] == 2:
            local_points = np.column_stack((local_points, np.zeros(local_points.shape[0])))
        return self._value_and_local_gradient(local_points)[0]

    def distance_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local_points = np.atleast_2d(points).astype(float)
        if local_points.shape[1] == 2:
            local_points = np.column_stack((local_points, np.zeros(local_points.shape[0])))
        values, gradients = self._value_and_local_gradient(local_points)
        return values, gradients[:, :2]

    def clearance_and_gradient(self, state: np.ndarray, field: ObstacleField) -> tuple[float, np.ndarray]:
        obstacle_points = self._obstacle_points(field)
        x, y = state[:2]
        theta = float(state[2]) if state.size >= 3 else 0.0
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        local_xy = (obstacle_points - np.array([x, y], dtype=float)) @ rot
        local_points = np.column_stack((local_xy, np.zeros(local_xy.shape[0])))
        values, local_gradients = self._value_and_local_gradient(local_points)
        nearest = int(np.argmin(values))
        value = float(values[nearest])

        gx, gy = local_gradients[nearest, 0], local_gradients[nearest, 1]
        world_gradient = np.array([-gx * c + gy * s, -gx * s - gy * c], dtype=float)
        norm = np.linalg.norm(world_gradient)
        if norm > 1e-8:
            world_gradient /= norm
        return value, world_gradient

    def _obstacle_points(self, field: ObstacleField) -> np.ndarray:
        cache_key = tuple((float(obs.center[0]), float(obs.center[1]), float(obs.radius)) for obs in field.obstacles)
        if cache_key != self._field_cache_key or self._field_cache_points is None:
            self._field_cache_key = cache_key
            self._field_cache_points = field.surface_points(self.points_per_obstacle)
        return self._field_cache_points

    def _value_and_local_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def _dense_params(params: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    layer = params[name]
    return np.asarray(layer["kernel"], dtype=float), np.asarray(layer["bias"], dtype=float)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0.0
    out = np.empty_like(x, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out
