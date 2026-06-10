from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class PlanarQuadrotorRobot(RobotModel):
    """Double-integrator x/y dynamics with single-integrator yaw."""

    def __init__(self) -> None:
        super().__init__(
            name="planar_quadrotor",
            state_dim=5,
            control_dim=3,
            control_bounds=jnp.array([[-1.1, 1.1], [-1.1, 1.1], [-2.2, 2.2]], dtype=float),
            radius=0.31,
            body_point_radius=0.06,
            default_state=jnp.array([-7.0, -3.6, -0.75, 0.0, 0.0], dtype=float),
            default_goal=jnp.array([7.0, 3.6], dtype=float),
            goal_tolerance=0.45,
        )
        self.body_length = 0.56
        self.body_width = 0.28
        self.velocity_limit = 1.35

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        ax_cmd, ay_cmd, yaw_rate = self.clip_control(control)
        x, y, yaw, vx, vy = state
        vx_next = jnp.clip(vx + ax_cmd * dt, -self.velocity_limit, self.velocity_limit)
        vy_next = jnp.clip(vy + ay_cmd * dt, -self.velocity_limit, self.velocity_limit)
        return jnp.array(
            [
                x + vx * dt,
                y + vy * dt,
                wrap_angle(yaw + yaw_rate * dt),
                vx_next,
                vy_next,
            ],
            dtype=float,
        )

    def drift(self, state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        return jnp.array([state[3], state[4], 0.0, 0.0, 0.0], dtype=float)

    def control_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )

    def projection_barrier_state(self, state: jnp.ndarray, dt: float) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        lookahead = 5.0 * dt
        return state.at[0].add(state[3] * lookahead).at[1].add(state[4] * lookahead)

    def nominal_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        position_error = jnp.asarray(goal, dtype=float) - self.position(state)
        desired_velocity = jnp.clip(0.85 * position_error, -self.velocity_limit, self.velocity_limit)
        accel = 1.4 * (desired_velocity - state[3:5])
        desired_yaw = jnp.arctan2(position_error[1], position_error[0])
        yaw_rate = 2.2 * wrap_angle(desired_yaw - state[2])
        return self.clip_control(jnp.array([accel[0], accel[1], yaw_rate], dtype=float))

    def body_points(self, state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        x, y, yaw = state[:3]
        rot = self._rot(yaw)
        body = jnp.array(
            [
                [-self.body_length / 2, -self.body_width / 2],
                [self.body_length / 2, -self.body_width / 2],
                [self.body_length / 2, self.body_width / 2],
                [-self.body_length / 2, self.body_width / 2],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        return body @ rot.T + jnp.array([x, y])

    def draw(self, ax, state: jnp.ndarray, **kwargs) -> None:
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        state_np = np.asarray(state, dtype=float)
        x, y, yaw = state_np[:3]
        rot = np.asarray(self._rot(yaw), dtype=float)
        local_body = np.array(
            [
                [-self.body_length / 2, -self.body_width / 2],
                [self.body_length / 2, -self.body_width / 2],
                [self.body_length / 2, self.body_width / 2],
                [-self.body_length / 2, self.body_width / 2],
            ],
            dtype=float,
        )
        body = local_body @ rot.T + np.array([x, y])
        ax.add_patch(Polygon(body, closed=True, facecolor=color, edgecolor=edgecolor, alpha=0.85, **kwargs))

    @staticmethod
    def _rot(theta) -> jnp.ndarray:
        return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
