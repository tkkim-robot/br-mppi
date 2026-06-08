from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


@dataclass
class RobotModel:
    name: str
    state_dim: int
    control_dim: int
    control_bounds: np.ndarray
    radius: float
    body_point_radius: float
    default_state: np.ndarray
    default_goal: np.ndarray
    goal_tolerance: float = 0.35

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError

    def drift(self, state: np.ndarray) -> np.ndarray:
        """Continuous-time drift f(x) for control-affine dynamics xdot = f(x) + g(x)u."""
        return np.zeros(self.state_dim, dtype=float)

    def control_matrix(self, state: np.ndarray) -> np.ndarray:
        """Continuous-time control matrix g(x) for control-affine dynamics."""
        raise NotImplementedError

    def projection_barrier_state(self, state: np.ndarray, dt: float) -> np.ndarray:
        """State used to evaluate BR-MPPI projection barriers."""
        return state.copy()

    def position(self, state: np.ndarray) -> np.ndarray:
        return state[:2].astype(float)

    def body_points(self, state: np.ndarray) -> np.ndarray:
        return self.position(state)[None, :]

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def clip_control(self, control: np.ndarray) -> np.ndarray:
        return np.clip(control, self.control_bounds[:, 0], self.control_bounds[:, 1])

    def clip_controls(self, controls: np.ndarray) -> np.ndarray:
        return np.clip(controls, self.control_bounds[:, 0], self.control_bounds[:, 1])

    def draw(self, ax, state: np.ndarray, **kwargs) -> None:
        ax.scatter([state[0]], [state[1]], s=90, **kwargs)
