from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class UnicycleRobot(RobotModel):
    def __init__(self) -> None:
        super().__init__(
            name="unicycle",
            state_dim=3,
            control_dim=2,
            control_bounds=np.array([[-0.45, 1.45], [-2.2, 2.2]], dtype=float),
            radius=0.54,
            body_point_radius=0.08,
            default_state=np.array([-7.0, -3.6, -0.75], dtype=float),
            default_goal=np.array([7.0, 3.6], dtype=float),
            goal_tolerance=0.45,
        )
        self.length = 1.0
        self.width = 0.4

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        v, omega = self.clip_control(control)
        theta = state[2]
        next_state = state.copy()
        next_state[0] += v * np.cos(theta) * dt
        next_state[1] += v * np.sin(theta) * dt
        next_state[2] = wrap_angle(theta + omega * dt)
        return next_state

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - self.position(state)
        desired = np.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired - state[2])
        distance = np.linalg.norm(delta)
        v = np.clip(0.9 * distance * max(0.15, np.cos(heading_error)), -0.2, 1.2)
        omega = 2.4 * heading_error
        return self.clip_control(np.array([v, omega], dtype=float))

    def body_points(self, state: np.ndarray) -> np.ndarray:
        x, y, theta = state
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
        points = self.body_points(state)[:4]
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=edgecolor, alpha=0.8, **kwargs))
