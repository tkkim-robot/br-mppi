from __future__ import annotations

import numpy as np

from robots.base_robot import RobotModel


class SingleIntegratorRobot(RobotModel):
    def __init__(self) -> None:
        super().__init__(
            name="single_integrator",
            state_dim=2,
            control_dim=2,
            control_bounds=np.array([[-1.25, 1.25], [-1.25, 1.25]], dtype=float),
            radius=0.22,
            body_point_radius=0.22,
            default_state=np.array([-7.0, -3.6], dtype=float),
            default_goal=np.array([7.0, 3.6], dtype=float),
        )

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        u = self.clip_control(control)
        return state + u * dt

    def control_matrix(self, state: np.ndarray) -> np.ndarray:
        return np.eye(2, dtype=float)

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return self.clip_control(0.85 * (goal - self.position(state)))
