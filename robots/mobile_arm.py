from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Polygon

from robots.base_robot import RobotModel, wrap_angle


class MobileArmRobot(RobotModel):
    """Planar mobile base with two fixed 4-link arms."""

    def __init__(self) -> None:
        default_state = jnp.array(
            [
                -8.2,
                -4.45,
                -0.55,
                jnp.pi / 2 + 0.3 * jnp.pi / 2,
                -jnp.pi / 6,
                -jnp.pi / 12,
                -jnp.pi / 12,
                jnp.pi / 2 - 0.3 * jnp.pi / 2,
                jnp.pi / 6,
                jnp.pi / 12,
                jnp.pi / 12,
            ],
            dtype=float,
        )
        super().__init__(
            name="mobile_arm",
            state_dim=11,
            control_dim=10,
            control_bounds=jnp.array(
                [
                    [-0.25, 1.65],
                    [-2.1, 2.1],
                    *[[-0.6, 0.6] for _ in range(8)],
                ],
                dtype=float,
            ),
            radius=0.2,
            body_point_radius=0.02,
            default_state=default_state,
            default_goal=jnp.array([8.2, 4.45], dtype=float),
            goal_tolerance=0.9,
        )
        self.base_length = 1.5
        self.base_width = 1.0
        self.arm_offsets = jnp.array([[-0.5, 0.0], [0.5, 0.0]], dtype=float)
        self.link_lengths = jnp.array([[0.8, 0.6, 0.4, 0.2], [0.8, 0.6, 0.4, 0.2]], dtype=float)
        self.link_widths = jnp.array([[0.2, 0.15, 0.1, 0.05], [0.2, 0.15, 0.1, 0.05]], dtype=float)
        self.base_edge_samples = 6
        self.link_edge_samples = 5
        self.rest_joints = default_state[3:]

    def step(self, state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        u = self.clip_control(control)
        v, omega = u[:2]
        theta = state[2]
        base_next = jnp.array(
            [
                state[0] + v * jnp.cos(theta) * dt,
                state[1] + v * jnp.sin(theta) * dt,
                wrap_angle(theta + omega * dt),
            ],
            dtype=float,
        )
        joints = wrap_angle(state[3:] + u[2:] * dt)
        return jnp.concatenate((base_next, joints))

    def control_matrix(self, state: jnp.ndarray) -> jnp.ndarray:
        theta = jnp.asarray(state, dtype=float)[2]
        matrix = jnp.zeros((self.state_dim, self.control_dim), dtype=float)
        matrix = matrix.at[0, 0].set(jnp.cos(theta))
        matrix = matrix.at[1, 0].set(jnp.sin(theta))
        matrix = matrix.at[2, 1].set(1.0)
        matrix = matrix.at[3:, 2:].set(jnp.eye(8, dtype=float))
        return matrix

    def nominal_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        delta = jnp.asarray(goal, dtype=float) - self.position(state)
        desired = jnp.arctan2(delta[1], delta[0])
        heading_error = wrap_angle(desired - state[2])
        distance = jnp.linalg.norm(delta)
        v = jnp.clip(0.7 * distance * jnp.maximum(0.1, jnp.cos(heading_error)), 0.0, 1.35)
        omega = 2.0 * heading_error
        joint_velocity = -0.35 * (state[3:] - self.rest_joints)
        return self.clip_control(jnp.concatenate((jnp.array([v, omega], dtype=float), joint_velocity)))

    def body_points(self, state: jnp.ndarray) -> jnp.ndarray:
        base = self.base_polygon(state)
        base_points = self._sample_polygon_edges(base, self.base_edge_samples)
        base_center = jnp.mean(base, axis=0, keepdims=True)
        link_polygons = self.link_polygons(state)
        link_samples = jax.vmap(lambda polygon: self._sample_polygon_edges(polygon, self.link_edge_samples))(link_polygons)
        link_centers = jnp.mean(link_polygons, axis=1, keepdims=True)
        per_link = jnp.concatenate((link_samples, link_centers), axis=1).reshape((-1, 2))
        return jnp.vstack((base_points, base_center, per_link))

    def base_polygon(self, state: jnp.ndarray) -> jnp.ndarray:
        base = jnp.array(
            [
                [-self.base_length / 2, -self.base_width / 2],
                [self.base_length / 2, -self.base_width / 2],
                [self.base_length / 2, self.base_width / 2],
                [-self.base_length / 2, self.base_width / 2],
            ],
            dtype=float,
        )
        state = jnp.asarray(state, dtype=float)
        x, y, theta = state[:3]
        rot = self._rot(theta)
        return base @ rot.T + jnp.array([x, y])

    def arm_polylines(self, state: jnp.ndarray) -> jnp.ndarray:
        state = jnp.asarray(state, dtype=float)
        x, y, theta = state[:3]
        rot = self._rot(theta)
        joints = state[3:].reshape(2, 4)
        shoulders = jnp.array([x, y]) + self.arm_offsets @ rot.T

        def arm_points(shoulder, arm_joints, lengths):
            angles = theta + jnp.cumsum(arm_joints)
            directions = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
            displacements = lengths[:, None] * directions
            endpoints = shoulder[None, :] + jnp.cumsum(displacements, axis=0)
            return jnp.vstack((shoulder[None, :], endpoints))

        return jax.vmap(arm_points)(shoulders, joints, self.link_lengths)

    def link_polygons(self, state: jnp.ndarray) -> jnp.ndarray:
        arms = self.arm_polylines(state)
        starts = arms[:, :-1, :]
        ends = arms[:, 1:, :]
        tangents = ends - starts
        norms = jnp.maximum(jnp.linalg.norm(tangents, axis=2), 1e-8)
        normals = jnp.stack((-tangents[:, :, 1], tangents[:, :, 0]), axis=2) / norms[:, :, None]
        offsets = 0.5 * self.link_widths[:, :, None] * normals
        polygons = jnp.stack((starts - offsets, ends - offsets, ends + offsets, starts + offsets), axis=2)
        return polygons.reshape((-1, 4, 2))

    def draw(self, ax, state: jnp.ndarray, **kwargs) -> None:
        color = kwargs.pop("color", "tab:orange")
        edgecolor = kwargs.pop("edgecolor", "black")
        ax.add_patch(
            Polygon(
                np.asarray(self.base_polygon(state), dtype=float),
                closed=True,
                facecolor=color,
                edgecolor=edgecolor,
                alpha=0.55,
                **kwargs,
            )
        )
        for polygon in np.asarray(self.link_polygons(state), dtype=float):
            ax.add_patch(
                Polygon(polygon, closed=True, facecolor="tab:orange", edgecolor=edgecolor, alpha=0.78, linewidth=0.8)
            )
        for arm in np.asarray(self.arm_polylines(state), dtype=float):
            ax.plot(arm[:, 0], arm[:, 1], linewidth=3.0, color="tab:orange", solid_capstyle="round")
            ax.scatter(arm[:-1, 0], arm[:-1, 1], s=26, color="white", edgecolor=edgecolor, zorder=5)
            ax.scatter(arm[-1:, 0], arm[-1:, 1], s=34, marker="s", color="tab:red", edgecolor=edgecolor, zorder=6)

    @staticmethod
    def _rot(theta) -> jnp.ndarray:
        return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

    @staticmethod
    def _sample_polygon_edges(polygon: jnp.ndarray, samples_per_edge: int) -> jnp.ndarray:
        count = max(2, int(samples_per_edge))
        starts = polygon
        ends = jnp.roll(polygon, -1, axis=0)
        t = jnp.linspace(0.0, 1.0, count, endpoint=False, dtype=polygon.dtype)
        samples = (1.0 - t[None, :, None]) * starts[:, None, :] + t[None, :, None] * ends[:, None, :]
        return samples.reshape((-1, 2))
