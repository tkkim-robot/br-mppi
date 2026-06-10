from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from matplotlib.patches import Circle


@dataclass(frozen=True)
class CircleObstacle:
    center: jnp.ndarray
    radius: float

    def __init__(self, center: tuple[float, float], radius: float) -> None:
        object.__setattr__(self, "center", jnp.array(center, dtype=float))
        object.__setattr__(self, "radius", float(radius))


@dataclass
class ObstacleField:
    obstacles: tuple[CircleObstacle, ...]

    @property
    def centers(self) -> jnp.ndarray:
        return jnp.stack([obs.center for obs in self.obstacles])

    @property
    def radii(self) -> jnp.ndarray:
        return jnp.array([obs.radius for obs in self.obstacles], dtype=float)

    def signed_distance(self, points: jnp.ndarray) -> jnp.ndarray:
        points = jnp.atleast_2d(jnp.asarray(points, dtype=float))
        distances = jnp.linalg.norm(points[:, None, :] - self.centers[None, :, :], axis=2) - self.radii[None, :]
        return jnp.min(distances, axis=1)

    def distance_and_gradient(self, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        points = jnp.atleast_2d(jnp.asarray(points, dtype=float))
        delta = points[:, None, :] - self.centers[None, :, :]
        norms = jnp.linalg.norm(delta, axis=2)
        safe_norms = jnp.maximum(norms, 1e-8)
        distances = norms - self.radii[None, :]
        nearest = jnp.argmin(distances, axis=1)
        chosen_distances = jnp.take_along_axis(distances, nearest[:, None], axis=1)[:, 0]
        gradients = delta / safe_norms[:, :, None]
        chosen_gradients = jnp.take_along_axis(gradients, nearest[:, None, None], axis=1)[:, 0, :]
        return chosen_distances, chosen_gradients

    def surface_points(self, points_per_obstacle: int = 48) -> jnp.ndarray:
        theta = jnp.linspace(0.0, 2.0 * jnp.pi, points_per_obstacle, endpoint=False)
        unit_circle = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=1)
        return (self.centers[:, None, :] + self.radii[:, None, None] * unit_circle[None, :, :]).reshape((-1, 2))

    def draw(self, ax) -> None:
        for obs in self.obstacles:
            ax.add_patch(
                Circle(
                    np.asarray(obs.center, dtype=float),
                    obs.radius,
                    facecolor="black",
                    edgecolor="black",
                    alpha=1.0,
                    linewidth=1.0,
                    zorder=8,
                )
            )


def default_obstacle_field(robot_name: str | None = None) -> ObstacleField:
    if robot_name == "mobile_arm":
        return ObstacleField(
            obstacles=(
                CircleObstacle(center=(-5.55, -0.25), radius=0.45),
                CircleObstacle(center=(-5.25, -5.75), radius=0.45),
                CircleObstacle(center=(-4.25, -2.65), radius=0.34),
                CircleObstacle(center=(-3.35, 1.65), radius=0.46),
                CircleObstacle(center=(-2.65, -3.65), radius=0.46),
                CircleObstacle(center=(-1.15, -0.35), radius=0.34),
                CircleObstacle(center=(-0.75, 2.75), radius=0.46),
                CircleObstacle(center=(0.0, -2.15), radius=0.45),
                CircleObstacle(center=(1.0, 4.45), radius=0.45),
                CircleObstacle(center=(1.85, 1.15), radius=0.34),
                CircleObstacle(center=(2.55, -0.95), radius=0.45),
                CircleObstacle(center=(4.15, 5.25), radius=0.42),
                CircleObstacle(center=(4.45, 2.55), radius=0.34),
                CircleObstacle(center=(4.95, -1.55), radius=0.45),
                CircleObstacle(center=(6.35, 2.2), radius=0.45),
                CircleObstacle(center=(6.65, -3.15), radius=0.45),
            )
        )
    return ObstacleField(
        obstacles=(
            CircleObstacle(center=(-6.15, -1.55), radius=0.50),
            CircleObstacle(center=(-5.35, -4.25), radius=0.55),
            CircleObstacle(center=(-4.55, 1.15), radius=0.52),
            CircleObstacle(center=(-3.55, -2.55), radius=0.48),
            CircleObstacle(center=(-3.05, 0.15), radius=0.50),
            CircleObstacle(center=(-2.05, 2.25), radius=0.52),
            CircleObstacle(center=(-1.45, -2.95), radius=0.54),
            CircleObstacle(center=(-0.55, 0.75), radius=0.62),
            CircleObstacle(center=(0.45, -1.75), radius=0.54),
            CircleObstacle(center=(1.15, 2.55), radius=0.52),
            CircleObstacle(center=(1.95, -0.35), radius=0.58),
            CircleObstacle(center=(2.7, 3.55), radius=0.46),
            CircleObstacle(center=(3.25, 1.35), radius=0.60),
            CircleObstacle(center=(3.95, -2.35), radius=0.50),
            CircleObstacle(center=(4.9, 4.35), radius=0.48),
            CircleObstacle(center=(5.1, 0.2), radius=0.56),
            CircleObstacle(center=(6.0, 1.1), radius=0.46),
            CircleObstacle(center=(6.05, -1.55), radius=0.48),
            CircleObstacle(center=(-0.15, 4.25), radius=0.44),
            CircleObstacle(center=(2.2, -4.0), radius=0.45),
        )
    )
