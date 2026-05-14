from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from robots.base_robot import RobotModel
from sdf.geometry import ObstacleField


ALGORITHMS = ("brmppi", "mppi", "penalty_mppi")


class SignedDistanceModel(Protocol):
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        ...

    def distance_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class MPPIConfig:
    horizon: int = 24
    samples: int = 192
    dt: float = 0.08
    temperature: float = 2.5
    noise_scale: float = 0.35
    goal_weight: float = 1.0
    final_goal_weight: float = 8.0
    control_weight: float = 0.03
    safety_weight: float = 18.0
    collision_weight: float = 3000.0
    barrier_rate_weight: float = 30.0
    barrier_alpha: float = 3.0
    barrier_trigger_distance: float = 1.1
    plot_samples: int = 100


class MPPIController:
    """Sampling MPPI controller with optional barrier-rate guidance."""

    def __init__(
        self,
        robot: RobotModel,
        obstacle_field: ObstacleField,
        *,
        algo: str = "brmppi",
        sdf_model: SignedDistanceModel | None = None,
        config: MPPIConfig | None = None,
        seed: int = 7,
    ) -> None:
        if algo not in ALGORITHMS:
            raise ValueError(f"Unknown MPPI algorithm '{algo}'. Choose one of {ALGORITHMS}.")
        self.robot = robot
        self.obstacle_field = obstacle_field
        self.sdf_model = sdf_model or obstacle_field
        self.algo = algo
        self.config = config or MPPIConfig()
        self.rng = np.random.default_rng(seed)
        self.control_sequence = np.zeros((self.config.horizon, robot.control_dim), dtype=float)
        self.last_diagnostics: dict[str, float | bool | np.ndarray] = {}

    def command(self, state: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict[str, float | bool | np.ndarray]]:
        nominal = self.robot.nominal_control(state, goal)
        if not np.any(self.control_sequence):
            self.control_sequence[:] = nominal

        candidates = self._sample_control_sequences(state, nominal)
        effective = np.zeros_like(candidates)
        trajectories = np.zeros((self.config.samples, self.config.horizon + 1, self.robot.state_dim), dtype=float)
        costs = np.zeros(self.config.samples, dtype=float)
        min_clearances = np.zeros(self.config.samples, dtype=float)

        for sample_idx in range(self.config.samples):
            rollout = self._rollout(state, goal, candidates[sample_idx])
            costs[sample_idx] = rollout["cost"]
            effective[sample_idx] = rollout["controls"]
            trajectories[sample_idx] = rollout["states"]
            min_clearances[sample_idx] = rollout["min_clearance"]

        weights = self._weights(costs)
        updated = np.einsum("s,shm->hm", weights, effective)
        updated = self.robot.clip_controls(updated)

        action = updated[0].copy()
        self.control_sequence[:-1] = updated[1:]
        self.control_sequence[-1] = nominal

        best_idx = int(np.argmin(costs))
        plot_count = min(self.config.plot_samples, self.config.samples)
        diagnostics: dict[str, float | bool | np.ndarray] = {
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            "best_min_clearance": float(min_clearances[best_idx]),
            "weighted_min_clearance": float(np.sum(weights * min_clearances)),
            "best_trajectory": trajectories[best_idx],
            "best_controls": effective[best_idx],
            "sampled_trajectories": trajectories[:plot_count],
        }
        self.last_diagnostics = diagnostics
        return action, diagnostics

    def _sample_control_sequences(self, state: np.ndarray, nominal: np.ndarray) -> np.ndarray:
        cfg = self.config
        bounds = self.robot.control_bounds
        sigma = cfg.noise_scale * (bounds[:, 1] - bounds[:, 0])
        noise = self.rng.normal(0.0, sigma, size=(cfg.samples, cfg.horizon, self.robot.control_dim))

        base = self.control_sequence.copy()
        base[-1] = nominal
        if cfg.horizon > 1:
            base = 0.72 * base + 0.28 * nominal

        candidates = base[None, :, :] + noise
        candidates[0] = base
        return self.robot.clip_controls(candidates)

    def _weights(self, costs: np.ndarray) -> np.ndarray:
        shifted = costs - np.min(costs)
        scaled = -shifted / max(self.config.temperature, 1e-6)
        scaled = np.clip(scaled, -60.0, 0.0)
        weights = np.exp(scaled)
        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            return np.ones_like(costs) / costs.size
        return weights / total

    def _rollout(self, state: np.ndarray, goal: np.ndarray, controls: np.ndarray) -> dict[str, np.ndarray | float]:
        cfg = self.config
        x = state.copy()
        states = np.zeros((cfg.horizon + 1, self.robot.state_dim), dtype=float)
        effective_controls = np.zeros_like(controls)
        states[0] = x
        cost = 0.0
        min_clearance = self._clearance(x)

        for t in range(cfg.horizon):
            u = controls[t].copy()
            h_now = self._clearance(x)

            if self.algo == "brmppi":
                u = self._barrier_rate_project(x, u)

            x_next = self.robot.step(x, u, cfg.dt)
            h_next = self._clearance(x_next)
            hdot = (h_next - h_now) / cfg.dt
            min_clearance = min(min_clearance, h_next)

            goal_error = np.linalg.norm(self.robot.position(x_next) - goal)
            cost += cfg.goal_weight * goal_error**2
            cost += cfg.control_weight * float(np.dot(u, u))

            if self.algo == "penalty_mppi":
                cost += self._safety_cost(h_next)
            elif self.algo == "brmppi":
                cost += self._safety_cost(h_next)
                violation = max(0.0, -hdot - cfg.barrier_alpha * h_now)
                cost += cfg.barrier_rate_weight * violation**2

            if self.algo != "mppi" and h_next < 0.0:
                cost += cfg.collision_weight * (1.0 + abs(h_next)) ** 2

            effective_controls[t] = u
            states[t + 1] = x_next
            x = x_next

        final_error = np.linalg.norm(self.robot.position(x) - goal)
        cost += cfg.final_goal_weight * final_error**2
        return {
            "cost": float(cost),
            "states": states,
            "controls": effective_controls,
            "min_clearance": float(min_clearance),
        }

    def _safety_cost(self, clearance: float) -> float:
        margin = 0.75
        if clearance >= margin:
            return 0.0
        if clearance <= 0.0:
            return self.config.safety_weight * (margin - clearance) ** 2 + self.config.collision_weight
        return self.config.safety_weight * (margin - clearance) ** 2

    def _clearance(self, state: np.ndarray) -> float:
        points = self.robot.body_points(state)
        distances = self.obstacle_field.signed_distance(points)
        return float(np.min(distances) - self.robot.body_point_radius)

    def _center_clearance_and_gradient(self, state: np.ndarray) -> tuple[float, np.ndarray]:
        if hasattr(self.sdf_model, "clearance_and_gradient"):
            return self.sdf_model.clearance_and_gradient(state, self.obstacle_field)
        return self._analytic_center_clearance_and_gradient(state)

    def _analytic_center_clearance_and_gradient(self, state: np.ndarray) -> tuple[float, np.ndarray]:
        position = self.robot.position(state)[None, :]
        distances, gradients = self.obstacle_field.distance_and_gradient(position)
        return float(distances[0] - self.robot.radius), gradients[0]

    def _barrier_rate_project(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        cfg = self.config
        if self.sdf_model is not self.obstacle_field:
            approximate_h, _ = self._analytic_center_clearance_and_gradient(state)
            if approximate_h > cfg.barrier_trigger_distance:
                return self.robot.clip_control(control)

        h, grad = self._center_clearance_and_gradient(state)
        if h > cfg.barrier_trigger_distance:
            return self.robot.clip_control(control)

        p = self.robot.position(state)
        zero = np.zeros(self.robot.control_dim, dtype=float)
        p_zero = self.robot.position(self.robot.step(state, zero, cfg.dt))
        drift_hdot = float(grad @ ((p_zero - p) / cfg.dt))

        jac = np.zeros(self.robot.control_dim, dtype=float)
        eps = 1e-4
        for i in range(self.robot.control_dim):
            du = np.zeros(self.robot.control_dim, dtype=float)
            du[i] = eps
            p_eps = self.robot.position(self.robot.step(state, du, cfg.dt))
            jac[i] = float(grad @ ((p_eps - p_zero) / (eps * cfg.dt)))

        target = -cfg.barrier_alpha * h - drift_hdot
        current = float(jac @ control)
        denom = float(jac @ jac) + 1e-9
        if current < target and denom > 1e-8:
            control = control + ((target - current) / denom) * jac
        return self.robot.clip_control(control)
