from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def wrap_angle(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


@dataclass
class RobotModel:
    name: str
    state_dim: int
    control_dim: int
    control_bounds: jnp.ndarray
    radius: float
    body_point_radius: float
    default_state: jnp.ndarray
    default_goal: jnp.ndarray
    goal_tolerance: float = 0.35

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        raise NotImplementedError

    def drift(self, state: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time drift f(x) for control-affine dynamics xdot = f(x) + g(x)u."""
        return jnp.zeros(self.state_dim, dtype=float)

    def control_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        """Continuous-time control matrix g(x) for control-affine dynamics."""
        raise NotImplementedError

    def projection_barrier_state(self, state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """State used to evaluate BR-MPPI projection barriers."""
        return jnp.asarray(state, dtype=float)

    def position(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(state, dtype=float)[:2]

    def body_points(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.position(state)[None, :]

    def nominal_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def clip_control(self, control: jnp.ndarray) -> jnp.ndarray:
        bounds = jnp.asarray(self.control_bounds, dtype=float)
        return jnp.clip(jnp.asarray(control, dtype=float), bounds[:, 0], bounds[:, 1])

    def clip_controls(self, controls: jnp.ndarray) -> jnp.ndarray:
        bounds = jnp.asarray(self.control_bounds, dtype=float)
        return jnp.clip(jnp.asarray(controls, dtype=float), bounds[:, 0], bounds[:, 1])

    def draw(self, ax: Any, state: jnp.ndarray, **kwargs) -> None:
        state_np = np.asarray(state, dtype=float)
        ax.scatter([state_np[0]], [state_np[1]], s=90, **kwargs)
