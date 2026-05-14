from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class DynamicUnicycleRobot(RobotModel):
    def __init__(self) -> None:
        super().__init__(
            name="dynamic_unicycle",
            state_dim=5,
            control_dim=2,
            control_bounds=np.array([[-1.4, 1.4], [-3.0, 3.0]], dtype=float),
            radius=0.54,
            body_point_radius=0.08,
            default_state=np.array([-7.0, -3.6, -0.75, 0.0, 0.0], dtype=float),
            default_goal=np.array([7.0, 3.6], dtype=float),
            goal_tolerance=0.5,
        )
        self.length = 1.0
        self.width = 0.4
        self.speed_bounds = np.array([-0.35, 1.55], dtype=float)
        self.omega_bounds = np.array([-2.4, 2.4], dtype=float)

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        accel, angular_accel = self.clip_control(control)
        x, y, theta, v, omega = state
        v_next = float(np.clip(v + accel * dt, self.speed_bounds[0], self.speed_bounds[1]))
        omega_next = float(np.clip(omega + angular_accel * dt, self.omega_bounds[0], self.omega_bounds[1]))
        theta_next = wrap_angle(theta + omega_next * dt)
        return np.array(
            [
                x + v_next * np.cos(theta_next) * dt,
                y + v_next * np.sin(theta_next) * dt,
                theta_next,
                v_next,
                omega_next,
            ],
            dtype=float,
        )

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - self.position(state)
        desired = np.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired - state[2])
        distance = np.linalg.norm(delta)
        desired_v = np.clip(0.75 * distance * max(0.1, np.cos(heading_error)), 0.0, 1.25)
        desired_omega = np.clip(2.2 * heading_error, -2.0, 2.0)
        accel = 1.8 * (desired_v - state[3])
        angular_accel = 2.0 * (desired_omega - state[4])
        return self.clip_control(np.array([accel, angular_accel], dtype=float))

    def body_points(self, state: np.ndarray) -> np.ndarray:
        x, y, theta = state[:3]
        corners = np.array(
            [
                [-self.length / 2, -self.width / 2],
                [self.length / 2, -self.width / 2],
                [self.length / 2, self.width / 2],
                [-self.length / 2, self.width / 2],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return corners @ rot.T + np.array([x, y])

    def draw(self, ax, state: np.ndarray, **kwargs) -> None:
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        ax.add_patch(Polygon(self.body_points(state)[:4], closed=True, facecolor=color, edgecolor=edgecolor, alpha=0.8, **kwargs))
