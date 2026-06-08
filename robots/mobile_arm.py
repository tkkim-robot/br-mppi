from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class MobileArmRobot(RobotModel):
    """Planar mobile base with two fixed 4-link arms."""

    def __init__(self) -> None:
        super().__init__(
            name="mobile_arm",
            state_dim=11,
            control_dim=10,
            control_bounds=np.array(
                [
                    [-0.25, 1.65],
                    [-2.1, 2.1],
                    *[[-0.6, 0.6] for _ in range(8)],
                ],
                dtype=float,
            ),
            radius=0.2,
            body_point_radius=0.02,
            default_state=np.array(
                [
                    -8.2,
                    -4.45,
                    -0.55,
                    np.pi / 2 + 0.3 * np.pi / 2,
                    -np.pi / 6,
                    -np.pi / 12,
                    -np.pi / 12,
                    np.pi / 2 - 0.3 * np.pi / 2,
                    np.pi / 6,
                    np.pi / 12,
                    np.pi / 12,
                ],
                dtype=float,
            ),
            default_goal=np.array([8.2, 4.45], dtype=float),
            goal_tolerance=0.9,
        )
        self.base_length = 1.5
        self.base_width = 1.0
        self.arm_offsets = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=float)
        self.link_lengths = np.array([[0.8, 0.6, 0.4, 0.2], [0.8, 0.6, 0.4, 0.2]], dtype=float)
        self.link_widths = np.array([[0.2, 0.15, 0.1, 0.05], [0.2, 0.15, 0.1, 0.05]], dtype=float)
        self.base_edge_samples = 6
        self.link_edge_samples = 5
        self.rest_joints = self.default_state[3:].copy()

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        u = self.clip_control(control)
        v, omega = u[:2]
        next_state = state.copy()
        theta = state[2]
        next_state[0] += v * np.cos(theta) * dt
        next_state[1] += v * np.sin(theta) * dt
        next_state[2] = wrap_angle(theta + omega * dt)
        joints = state[3:] + u[2:] * dt
        next_state[3:] = np.arctan2(np.sin(joints), np.cos(joints))
        return next_state

    def control_matrix(self, state: np.ndarray) -> np.ndarray:
        theta = state[2]
        matrix = np.zeros((self.state_dim, self.control_dim), dtype=float)
        matrix[0, 0] = np.cos(theta)
        matrix[1, 0] = np.sin(theta)
        matrix[2, 1] = 1.0
        matrix[3:, 2:] = np.eye(8)
        return matrix

    def nominal_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        delta = goal - self.position(state)
        desired = np.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired - state[2])
        distance = np.linalg.norm(delta)
        v = np.clip(0.7 * distance * max(0.1, np.cos(heading_error)), 0.0, 1.35)
        omega = 2.0 * heading_error
        joint_velocity = -0.35 * (state[3:] - self.rest_joints)
        return self.clip_control(np.concatenate((np.array([v, omega], dtype=float), joint_velocity)))

    def body_points(self, state: np.ndarray) -> np.ndarray:
        base = self.base_polygon(state)
        points = [self._sample_polygon_edges(base, self.base_edge_samples), np.mean(base, axis=0, keepdims=True)]
        for polygon in self.link_polygons(state):
            points.append(self._sample_polygon_edges(polygon, self.link_edge_samples))
            points.append(np.mean(polygon, axis=0, keepdims=True))
        return np.vstack(points)

    def base_polygon(self, state: np.ndarray) -> np.ndarray:
        base = np.array(
            [
                [-self.base_length / 2, -self.base_width / 2],
                [self.base_length / 2, -self.base_width / 2],
                [self.base_length / 2, self.base_width / 2],
                [-self.base_length / 2, self.base_width / 2],
            ],
            dtype=float,
        )
        x, y, theta = state[:3]
        rot = self._rot(theta)
        return base @ rot.T + np.array([x, y])

    def arm_polylines(self, state: np.ndarray) -> list[np.ndarray]:
        x, y, theta = state[:3]
        rot = self._rot(theta)
        joints = state[3:].reshape(2, 4)
        arms = []
        for arm_idx in range(2):
            shoulder = np.array([x, y]) + rot @ self.arm_offsets[arm_idx]
            angle = theta
            points = [shoulder]
            start = shoulder
            for link_idx in range(4):
                angle += joints[arm_idx, link_idx]
                end = start + self.link_lengths[arm_idx, link_idx] * np.array([np.cos(angle), np.sin(angle)])
                points.append(end)
                start = end
            arms.append(np.vstack(points))
        return arms

    def link_polygons(self, state: np.ndarray) -> list[np.ndarray]:
        polygons = []
        for arm_idx, arm in enumerate(self.arm_polylines(state)):
            for link_idx, (start, end) in enumerate(zip(arm[:-1], arm[1:])):
                tangent = end - start
                norm = np.linalg.norm(tangent)
                if norm < 1e-8:
                    continue
                normal = np.array([-tangent[1], tangent[0]]) / norm
                half_width = 0.5 * self.link_widths[arm_idx, link_idx]
                offset = half_width * normal
                polygons.append(np.vstack((start - offset, end - offset, end + offset, start + offset)))
        return polygons

    def draw(self, ax, state: np.ndarray, **kwargs) -> None:
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        ax.add_patch(Polygon(self.base_polygon(state), closed=True, facecolor=color, edgecolor=edgecolor, alpha=0.55, **kwargs))
        for polygon in self.link_polygons(state):
            ax.add_patch(
                Polygon(polygon, closed=True, facecolor="tab:orange", edgecolor=edgecolor, alpha=0.78, linewidth=0.8)
            )
        for arm in self.arm_polylines(state):
            ax.plot(arm[:, 0], arm[:, 1], linewidth=3.0, color="tab:orange", solid_capstyle="round")
            ax.scatter(arm[:-1, 0], arm[:-1, 1], s=26, color="white", edgecolor=edgecolor, zorder=5)
            ax.scatter(arm[-1:, 0], arm[-1:, 1], s=34, marker="s", color="tab:red", edgecolor=edgecolor, zorder=6)

    @staticmethod
    def _rot(theta: float) -> np.ndarray:
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    @staticmethod
    def _sample_polygon_edges(polygon: np.ndarray, samples_per_edge: int) -> np.ndarray:
        samples = []
        count = max(2, int(samples_per_edge))
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            t = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
            samples.append((1.0 - t) * start + t * end)
        return np.vstack(samples)
