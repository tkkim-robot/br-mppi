from __future__ import annotations

import jax.numpy as jnp

from robots.base_robot import RobotModel


class SingleIntegratorRobot(RobotModel):
    def __init__(self) -> None:
        super().__init__(
            name="single_integrator",
            state_dim=2,
            control_dim=2,
            control_bounds=jnp.array([[-1.25, 1.25], [-1.25, 1.25]], dtype=float),
            radius=0.22,
            body_point_radius=0.22,
            default_state=jnp.array([-7.0, -3.6], dtype=float),
            default_goal=jnp.array([7.0, 3.6], dtype=float),
        )

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        u = self.clip_control(control)
        return jnp.asarray(state, dtype=float) + u * dt

    def control_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(2, dtype=float)

    def nominal_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        return self.clip_control(0.85 * (jnp.asarray(goal, dtype=float) - self.position(state)))
