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
    barrier_alpha: float = 1.0
    alpha_rate_bound: float = 10.0
    alpha_state_bound: float = 2.0
    alpha_noise_scale: float = 0.65
    alpha_projection_inverse_weight: float = 0.1
    bound_penalty: float = 0.0
    barrier_buffer_distance: float = 0.2
    barrier_projection_margin: float = 0.12
    barrier_alpha_cost_weight: float = 0.01
    plot_samples: int = 100


@dataclass(frozen=True)
class RolloutResult:
    cost: float
    states: np.ndarray
    controls: np.ndarray
    augmented_controls: np.ndarray
    alpha_trace: np.ndarray
    min_clearance: float


@dataclass(frozen=True)
class ProjectionProblem:
    physical_desired: np.ndarray
    alpha_desired: np.ndarray
    z_desired: np.ndarray
    a_matrix: np.ndarray
    b_vector: np.ndarray
    inverse_weight: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray


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
        self.num_barriers = len(obstacle_field.obstacles)
        self.augmented_control_dim = robot.control_dim + self.num_barriers if algo == "brmppi" else robot.control_dim
        self.control_sequence = np.zeros((self.config.horizon, self.augmented_control_dim), dtype=float)
        self.alpha_state = np.full(self.num_barriers, self.config.barrier_alpha, dtype=float)
        self.last_diagnostics: dict[str, object] = {}

    @property
    def barrier_source(self) -> str:
        if hasattr(self.sdf_model, "obstacle_barriers"):
            return "neural_sdf"
        return "analytic_sdf"

    def command(self, state: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        nominal = self._nominal_augmented_control(state, goal)
        if not np.any(self.control_sequence):
            self.control_sequence[:] = nominal

        candidates = self._sample_control_sequences(state, nominal)
        effective_physical = np.zeros((self.config.samples, self.config.horizon, self.robot.control_dim), dtype=float)
        effective_augmented = np.zeros_like(candidates)
        alpha_traces = np.zeros((self.config.samples, self.config.horizon + 1, self.num_barriers), dtype=float)
        trajectories = np.zeros((self.config.samples, self.config.horizon + 1, self.robot.state_dim), dtype=float)
        costs = np.zeros(self.config.samples, dtype=float)
        min_clearances = np.zeros(self.config.samples, dtype=float)

        for sample_idx in range(self.config.samples):
            rollout = self._rollout(state, goal, candidates[sample_idx])
            costs[sample_idx] = rollout.cost
            effective_physical[sample_idx] = rollout.controls
            effective_augmented[sample_idx] = rollout.augmented_controls
            alpha_traces[sample_idx] = rollout.alpha_trace
            trajectories[sample_idx] = rollout.states
            min_clearances[sample_idx] = rollout.min_clearance

        weights = self._weights(costs)
        if self.algo == "brmppi":
            updated = np.einsum("s,shm->hm", weights, candidates)
            updated = self._clip_augmented_controls(updated)
            action, alpha_next, projected_first = self._project_augmented_control(state, updated[0], self.alpha_state)
            self.alpha_state = alpha_next.copy()
        else:
            updated = np.einsum("s,shm->hm", weights, effective_physical)
            updated = self.robot.clip_controls(updated)
            action = updated[0].copy()
            projected_first = updated[0].copy()

        self.control_sequence[:-1] = updated[1:]
        self.control_sequence[-1] = updated[-1]

        best_idx = int(np.argmin(costs))
        plot_count = min(self.config.plot_samples, self.config.samples)
        diagnostics: dict[str, object] = {
            "barrier_source": self.barrier_source,
            "best_cost": float(costs[best_idx]),
            "mean_cost": float(np.mean(costs)),
            **self._rollout_safety_diagnostics(min_clearances, weights, best_idx),
            "sampled_costs": costs.copy(),
            "sampled_min_clearances": min_clearances.copy(),
            "best_trajectory": trajectories[best_idx],
            "best_controls": effective_physical[best_idx],
            "best_augmented_controls": effective_augmented[best_idx],
            "best_alpha_trace": alpha_traces[best_idx],
            "weighted_alpha_terminal": np.einsum("s,sn->n", weights, alpha_traces[:, -1, :]) if self.algo == "brmppi" else np.array([]),
            "projected_first_augmented": projected_first,
            "sampled_trajectories": trajectories[:plot_count],
        }
        self.last_diagnostics = diagnostics
        return action, diagnostics

    def _nominal_augmented_control(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        physical = self.robot.nominal_control(state, goal)
        if self.algo != "brmppi":
            return physical
        alpha_rate = -self.config.alpha_rate_bound * np.ones(self.num_barriers, dtype=float)
        return np.concatenate((physical, alpha_rate))

    def _sample_control_sequences(self, state: np.ndarray, _nominal: np.ndarray) -> np.ndarray:
        cfg = self.config
        if self.algo == "brmppi":
            bounds = self._augmented_control_bounds()
            physical_sigma = cfg.noise_scale * (self.robot.control_bounds[:, 1] - self.robot.control_bounds[:, 0])
            alpha_sigma = cfg.alpha_noise_scale * (bounds[self.robot.control_dim :, 1] - bounds[self.robot.control_dim :, 0])
            sigma = np.concatenate((physical_sigma, alpha_sigma))
        else:
            bounds = self.robot.control_bounds
            sigma = cfg.noise_scale * (bounds[:, 1] - bounds[:, 0])
        noise = self.rng.normal(0.0, sigma, size=(cfg.samples, cfg.horizon, self.augmented_control_dim))

        base = self.control_sequence.copy()
        candidates = base[None, :, :] + noise
        candidates[0] = base
        if self.algo == "brmppi":
            return self._clip_augmented_controls(candidates)
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

    def _rollout(self, state: np.ndarray, goal: np.ndarray, controls: np.ndarray) -> RolloutResult:
        cfg = self.config
        x = state.copy()
        alpha = self.alpha_state.copy()
        states = np.zeros((cfg.horizon + 1, self.robot.state_dim), dtype=float)
        alpha_trace = np.zeros((cfg.horizon + 1, self.num_barriers), dtype=float)
        effective_controls = np.zeros((cfg.horizon, self.robot.control_dim), dtype=float)
        effective_augmented = np.zeros((cfg.horizon, self.augmented_control_dim), dtype=float)
        states[0] = x
        if self.algo == "brmppi":
            alpha_trace[0] = alpha
        cost = 0.0
        min_clearance = self._clearance(x)

        for t in range(cfg.horizon):
            u = controls[t].copy()
            if self.algo == "brmppi":
                u, alpha, projected_augmented = self._project_augmented_control(x, u, alpha)
                effective_augmented[t] = projected_augmented
                alpha_trace[t + 1] = alpha

            x_next = self.robot.step(x, u, cfg.dt)
            h_next = self._clearance(x_next)
            min_clearance = min(min_clearance, h_next)

            goal_error = np.linalg.norm(self.robot.position(x_next) - goal)
            cost += cfg.goal_weight * goal_error**2
            cost += cfg.control_weight * float(np.dot(u, u))

            if self.algo == "penalty_mppi":
                cost += self._safety_cost(h_next)
            elif self.algo == "brmppi":
                cost += self._barrier_alpha_cost(x, alpha)

            if self.algo == "penalty_mppi" and h_next < 0.0:
                cost += cfg.collision_weight * (1.0 + abs(h_next)) ** 2

            effective_controls[t] = u
            if self.algo != "brmppi":
                effective_augmented[t] = u
            states[t + 1] = x_next
            x = x_next

        final_error = np.linalg.norm(self.robot.position(x) - goal)
        cost += cfg.final_goal_weight * final_error**2
        return RolloutResult(
            cost=float(cost),
            states=states,
            controls=effective_controls,
            augmented_controls=effective_augmented,
            alpha_trace=alpha_trace,
            min_clearance=float(min_clearance),
        )

    def _rollout_safety_diagnostics(
        self,
        min_clearances: np.ndarray,
        weights: np.ndarray,
        best_idx: int,
    ) -> dict[str, float | bool | int]:
        sampled_collision_count = int(np.count_nonzero(min_clearances < 0.0))
        return {
            "best_min_clearance": float(min_clearances[best_idx]),
            "best_collision": bool(min_clearances[best_idx] < 0.0),
            "sampled_min_clearance": float(np.min(min_clearances)),
            "sampled_collision_count": sampled_collision_count,
            "sampled_collision_fraction": float(sampled_collision_count / self.config.samples),
            "weighted_min_clearance": float(np.sum(weights * min_clearances)),
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

    def _barrier_values(self, state: np.ndarray) -> np.ndarray:
        if hasattr(self.sdf_model, "obstacle_barriers"):
            return np.asarray(self.sdf_model.obstacle_barriers(state, self.obstacle_field), dtype=float)
        return self._analytic_obstacle_barriers(state)

    def _analytic_obstacle_barriers(self, state: np.ndarray) -> np.ndarray:
        points = self.robot.body_points(state)
        centers = np.vstack([obs.center for obs in self.obstacle_field.obstacles])
        radii = np.array([obs.radius for obs in self.obstacle_field.obstacles])
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii[None, :]
        return np.min(distances, axis=0) - self.robot.body_point_radius

    def _barrier_alpha_cost(self, state: np.ndarray, alpha: np.ndarray) -> float:
        cfg = self.config
        h = self._projection_barrier_values(state)
        if h.size == 0:
            return 0.0
        hmin_idx = int(np.argmin(h))
        hmin = float(h[hmin_idx])
        if hmin <= 0.0 or hmin >= cfg.barrier_buffer_distance:
            return 0.0
        return float(cfg.barrier_alpha_cost_weight * alpha[hmin_idx] / max(hmin, 0.01))

    def _project_augmented_control(
        self,
        state: np.ndarray,
        augmented_control: np.ndarray,
        alpha: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        problem = self._projection_problem(state, augmented_control, alpha)
        if problem.a_matrix.size == 0:
            return problem.physical_desired, problem.alpha_desired, problem.z_desired

        projected = self._closed_form_projection(
            problem.z_desired,
            problem.a_matrix,
            problem.b_vector,
            problem.inverse_weight,
            problem.lower_bound,
            problem.upper_bound,
        )
        projected = self._repair_projected_control_bounds(projected, problem.a_matrix, problem.b_vector)
        return (
            projected[: self.robot.control_dim].copy(),
            projected[self.robot.control_dim :].copy(),
            projected,
        )

    def _projection_problem(
        self,
        state: np.ndarray,
        augmented_control: np.ndarray,
        alpha: np.ndarray,
    ) -> ProjectionProblem:
        cfg = self.config
        physical_desired = self.robot.clip_control(augmented_control[: self.robot.control_dim])
        alpha_rate = np.clip(
            augmented_control[self.robot.control_dim :],
            -cfg.alpha_rate_bound,
            cfg.alpha_rate_bound,
        )
        alpha_desired = np.clip(
            alpha + alpha_rate * cfg.dt,
            -cfg.alpha_state_bound,
            cfg.alpha_state_bound,
        )
        z_desired = np.concatenate((physical_desired, alpha_desired))
        h = self._projection_barrier_values_with_margin(state)
        if h.size == 0:
            empty_rows = np.zeros((0, self.augmented_control_dim), dtype=float)
            lower, upper = self._projection_bounds()
            return ProjectionProblem(
                physical_desired=physical_desired,
                alpha_desired=alpha_desired,
                z_desired=z_desired,
                a_matrix=empty_rows,
                b_vector=np.zeros(0, dtype=float),
                inverse_weight=np.ones(self.augmented_control_dim, dtype=float),
                lower_bound=lower,
                upper_bound=upper,
            )

        barrier_jacobian = self._projection_barrier_jacobian(state)
        drift = self.robot.drift(state)
        control_matrix = self.robot.control_matrix(state)
        a_matrix = np.zeros((h.size, self.augmented_control_dim), dtype=float)
        a_matrix[:, : self.robot.control_dim] = barrier_jacobian @ control_matrix
        a_matrix[:, self.robot.control_dim :] = np.diag(h)
        b_vector = -(barrier_jacobian @ drift)
        inverse_weight = np.concatenate(
            (
                np.ones(self.robot.control_dim, dtype=float),
                cfg.alpha_projection_inverse_weight * np.ones(self.num_barriers, dtype=float),
            )
        )
        lower, upper = self._projection_bounds()
        return ProjectionProblem(
            physical_desired=physical_desired,
            alpha_desired=alpha_desired,
            z_desired=z_desired,
            a_matrix=a_matrix,
            b_vector=b_vector,
            inverse_weight=inverse_weight,
            lower_bound=lower,
            upper_bound=upper,
        )

    def _projection_barrier_jacobian(self, state: np.ndarray) -> np.ndarray:
        h_base = self._projection_barrier_values_with_margin(state)
        jacobian = np.zeros((h_base.size, self.robot.state_dim), dtype=float)
        eps = 1e-4
        for state_idx in range(self.robot.state_dim):
            perturbed = state.copy()
            perturbed[state_idx] += eps
            h_eps = self._projection_barrier_values_with_margin(perturbed)
            jacobian[:, state_idx] = (h_eps - h_base) / eps
        return jacobian

    def _projection_barrier_values(self, state: np.ndarray) -> np.ndarray:
        barrier_state = self.robot.projection_barrier_state(state, self.config.dt)
        return self._barrier_values(barrier_state)

    def _projection_barrier_values_with_margin(self, state: np.ndarray) -> np.ndarray:
        return self._projection_barrier_values(state) - self.config.barrier_projection_margin

    def _closed_form_projection(
        self,
        z_des: np.ndarray,
        a_matrix: np.ndarray,
        b_vector: np.ndarray,
        inverse_weight_diag: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        if a_matrix.size == 0:
            return z_des.copy()

        rho = max(0.0, self.config.bound_penalty)
        if rho > 0.0:
            weight_diag = 1.0 / np.maximum(inverse_weight_diag, 1e-9)
            m_diag = weight_diag + rho
            m_inv_diag = 1.0 / m_diag
            p = weight_diag * z_des + 0.5 * rho * (lower + upper)
            am_inv = a_matrix * m_inv_diag[None, :]
            rhs = am_inv @ p - b_vector
            correction = self._solve_projection_system(a_matrix, rhs, m_inv_diag)
            return m_inv_diag * (p - a_matrix.T @ correction)

        return self._unbounded_closed_form_projection(z_des, a_matrix, b_vector, inverse_weight_diag)

    def _unbounded_closed_form_projection(
        self,
        z_des: np.ndarray,
        a_matrix: np.ndarray,
        b_vector: np.ndarray,
        inverse_weight_diag: np.ndarray,
    ) -> np.ndarray:
        if a_matrix.size == 0:
            return z_des.copy()
        weighted_a_t = inverse_weight_diag[:, None] * a_matrix.T
        rhs = b_vector - a_matrix @ z_des
        lagrange = self._solve_projection_system(a_matrix, rhs, inverse_weight_diag)
        return z_des + weighted_a_t @ lagrange

    def _solve_projection_system(
        self,
        a_matrix: np.ndarray,
        rhs: np.ndarray,
        inverse_diag: np.ndarray,
    ) -> np.ndarray:
        if self._has_single_alpha_per_row(a_matrix):
            return self._solve_diagonal_plus_low_rank(a_matrix, rhs, inverse_diag)

        lhs = a_matrix @ (inverse_diag[:, None] * a_matrix.T)
        try:
            return np.linalg.solve(lhs + 1e-8 * np.eye(lhs.shape[0]), rhs)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(lhs) @ rhs

    def _has_single_alpha_per_row(self, a_matrix: np.ndarray) -> bool:
        control_dim = self.robot.control_dim
        if self.algo != "brmppi":
            return False
        if a_matrix.shape[1] != control_dim + self.num_barriers:
            return False
        alpha_block = a_matrix[:, control_dim:]
        return bool(np.all(np.count_nonzero(np.abs(alpha_block) > 1e-12, axis=1) <= 1))

    def _solve_diagonal_plus_low_rank(
        self,
        a_matrix: np.ndarray,
        rhs: np.ndarray,
        inverse_diag: np.ndarray,
    ) -> np.ndarray:
        control_dim = self.robot.control_dim
        au = a_matrix[:, :control_dim]
        alpha_block = a_matrix[:, control_dim:]
        inv_u = np.maximum(inverse_diag[:control_dim], 1e-12)
        inv_alpha = np.maximum(inverse_diag[control_dim:], 1e-12)
        diag_terms = (alpha_block**2) @ inv_alpha + 1e-8
        diag_inv = 1.0 / diag_terms
        diag_inv_au = diag_inv[:, None] * au
        small_lhs = np.diag(1.0 / inv_u) + au.T @ diag_inv_au
        small_rhs = au.T @ (diag_inv * rhs)
        try:
            small_solution = np.linalg.solve(small_lhs, small_rhs)
        except np.linalg.LinAlgError:
            small_solution = np.linalg.pinv(small_lhs) @ small_rhs
        return diag_inv * rhs - diag_inv_au @ small_solution

    def _repair_projected_control_bounds(
        self,
        projected: np.ndarray,
        a_matrix: np.ndarray,
        b_vector: np.ndarray,
    ) -> np.ndarray:
        control_dim = self.robot.control_dim
        control_lower = self.robot.control_bounds[:, 0]
        control_upper = self.robot.control_bounds[:, 1]
        bounded_control = np.clip(projected[:control_dim], control_lower, control_upper)
        if np.allclose(bounded_control, projected[:control_dim], rtol=0.0, atol=1e-10):
            return projected

        repaired = projected.copy()
        repaired[:control_dim] = bounded_control
        alpha_block = a_matrix[:, control_dim:]
        alpha_rhs = b_vector - a_matrix[:, :control_dim] @ bounded_control
        repaired_alpha = repaired[control_dim:]
        for row_idx, row in enumerate(alpha_block):
            alpha_columns = np.flatnonzero(np.abs(row) > 1e-8)
            if alpha_columns.size != 1:
                continue
            alpha_col = int(alpha_columns[0])
            repaired_alpha[alpha_col] = alpha_rhs[row_idx] / row[alpha_col]
        return repaired

    def _augmented_control_bounds(self) -> np.ndarray:
        alpha_bounds = np.column_stack(
            (
                -self.config.alpha_rate_bound * np.ones(self.num_barriers),
                self.config.alpha_rate_bound * np.ones(self.num_barriers),
            )
        )
        return np.vstack((self.robot.control_bounds, alpha_bounds))

    def _projection_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.concatenate(
            (
                self.robot.control_bounds[:, 0],
                -self.config.alpha_state_bound * np.ones(self.num_barriers),
            )
        )
        upper = np.concatenate(
            (
                self.robot.control_bounds[:, 1],
                self.config.alpha_state_bound * np.ones(self.num_barriers),
            )
        )
        return lower, upper

    def _clip_augmented_controls(self, controls: np.ndarray) -> np.ndarray:
        bounds = self._augmented_control_bounds()
        return np.clip(controls, bounds[:, 0], bounds[:, 1])
