from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class UnicycleRobot(RobotModel):
    def __init__(self) -> None:
        super().__init__(
            name="unicycle",
            state_dim=3,
            control_dim=2,
            control_bounds=jnp.array([[-0.45, 1.45], [-2.2, 2.2]], dtype=float),
            radius=0.54,
            body_point_radius=0.08,
            default_state=jnp.array([-7.0, -3.6, -0.75], dtype=float),
            default_goal=jnp.array([7.0, 3.6], dtype=float),
            goal_tolerance=0.45,
        )
        self.length = 1.0
        self.width = 0.4

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        v, omega = self.clip_control(control)
        theta = state[2]
        return jnp.array(
            [
                state[0] + v * jnp.cos(theta) * dt,
                state[1] + v * jnp.sin(theta) * dt,
                wrap_angle(theta + omega * dt),
            ],
            dtype=float,
        )

    def control_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.asarray(state, dtype=float)[2]
        return jnp.array(
            [
                [jnp.cos(theta), 0.0],
                [jnp.sin(theta), 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )

    def nominal_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        delta = jnp.asarray(goal, dtype=float) - self.position(state)
        desired = jnp.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired - state[2])
        distance = jnp.linalg.norm(delta)
        v = jnp.clip(0.9 * distance * jnp.maximum(0.15, jnp.cos(heading_error)), -0.2, 1.2)
        omega = 2.4 * heading_error
        return self.clip_control(jnp.array([v, omega], dtype=float))

    def body_points(self, state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        x, y, theta = state
        corners = jnp.array(
            [
                [-self.length / 2, -self.width / 2],
                [self.length / 2, -self.width / 2],
                [self.length / 2, self.width / 2],
                [-self.length / 2, self.width / 2],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        rot = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
        return corners @ rot.T + jnp.array([x, y])

    def draw(self, ax, state: jnp.ndarray, **kwargs) -> None:
        points = np.asarray(self.body_points(state)[:4], dtype=float)
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=edgecolor, alpha=0.8, **kwargs))
