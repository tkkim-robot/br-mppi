from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class DynamicUnicycleRobot(RobotModel):
    def __init__(self) -> None:
        super().__init__(
            name="dynamic_unicycle",
            state_dim=4,
            control_dim=2,
            control_bounds=jnp.array([[-3.0, 3.0], [-3.0, 3.0]], dtype=float),
            radius=0.54,
            body_point_radius=0.08,
            default_state=jnp.array([-7.0, -3.6, -0.75, 0.0], dtype=float),
            default_goal=jnp.array([7.0, 3.6], dtype=float),
            goal_tolerance=0.5,
        )
        self.length = 1.0
        self.width = 0.4

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        accel, omega = self.clip_control(control)
        x, y, theta, v = state
        return jnp.array(
            [
                x + v * jnp.cos(theta) * dt,
                y + v * jnp.sin(theta) * dt,
                wrap_angle(theta + omega * dt),
                v + accel * dt,
            ],
            dtype=float,
        )

    def drift(self, state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        _, _, theta, v = state
        return jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), 0.0, 0.0], dtype=float)

    def control_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )

    def projection_barrier_state(self, state: jnp.ndarray, dt: float) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        lookahead = 5.0 * dt
        theta, v = state[2], state[3]
        return state.at[0].add(v * jnp.cos(theta) * lookahead).at[1].add(v * jnp.sin(theta) * lookahead)

    def nominal_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        delta = jnp.asarray(goal, dtype=float) - self.position(state)
        desired = jnp.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired - state[2])
        distance = jnp.linalg.norm(delta)
        desired_v = jnp.clip(0.75 * distance * jnp.maximum(0.1, jnp.cos(heading_error)), 0.0, 1.25)
        accel = 1.8 * (desired_v - state[3])
        omega = 2.2 * heading_error
        return self.clip_control(jnp.array([accel, omega], dtype=float))

    def body_points(self, state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        x, y, theta = state[:3]
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
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        points = np.asarray(self.body_points(state)[:4], dtype=float)
        ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=edgecolor, alpha=0.8, **kwargs))
