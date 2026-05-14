from __future__ import annotations

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
            control_bounds=np.array([[-1.1, 1.1], [-1.1, 1.1], [-2.2, 2.2]], dtype=float),
            radius=0.31,
            body_point_radius=0.06,
            default_state=np.array([-7.0, -3.6, -0.75, 0.0, 0.0], dtype=float),
            default_goal=np.array([7.0, 3.6], dtype=float),
            goal_tolerance=0.45,
        )
        self.body_length = 0.56
        self.body_width = 0.28
        self.velocity_limit = 1.35

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        ax_cmd, ay_cmd, yaw_rate = self.clip_control(control)
        x, y, yaw, vx, vy = state
        vx_next = float(np.clip(vx + ax_cmd * dt, -self.velocity_limit, self.velocity_limit))
        vy_next = float(np.clip(vy + ay_cmd * dt, -self.velocity_limit, self.velocity_limit))
        return np.array(
            [
                x + vx_next * dt,
                y + vy_next * dt,
                wrap_angle(yaw + yaw_rate * dt),
                vx_next,
                vy_next,
            ],
            dtype=float,
        )

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        position_error = goal - self.position(state)
        desired_velocity = np.clip(0.85 * position_error, -self.velocity_limit, self.velocity_limit)
        accel = 1.4 * (desired_velocity - state[3:5])
        desired_yaw = np.arctan2(position_error[1], position_error[0])
        yaw_rate = 2.2 * wrap_angle(desired_yaw - state[2])
        return self.clip_control(np.array([accel[0], accel[1], yaw_rate], dtype=float))

    def body_points(self, state: np.ndarray) -> np.ndarray:
        x, y, yaw = state[:3]
        rot = self._rot(yaw)
        body = np.array(
            [
                [-self.body_length / 2, -self.body_width / 2],
                [self.body_length / 2, -self.body_width / 2],
                [self.body_length / 2, self.body_width / 2],
                [-self.body_length / 2, self.body_width / 2],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        return body @ rot.T + np.array([x, y])

    def draw(self, ax, state: np.ndarray, **kwargs) -> None:
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        x, y, yaw = state[:3]
        rot = self._rot(yaw)
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
    def _rot(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
