from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from robots.base_robot import RobotModel
from sdf.geometry import ObstacleField


ALGORITHMS = ("brmppi", "mppi", "penalty_mppi")


class SignedDistanceModel(Protocol):
    def signed_distance(self, points: jnp.ndarray) -> jnp.ndarray:
        ...

    def distance_and_gradient(self, points: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    br_clearance_margin: float = 0.45
    br_clearance_weight: float = 150.0
    br_collision_weight: float = 100000.0
    plot_samples: int = 100


@dataclass(frozen=True)
class RolloutResult:
    cost: jnp.ndarray
    states: jnp.ndarray
    controls: jnp.ndarray
    augmented_controls: jnp.ndarray
    alpha_trace: jnp.ndarray
    min_clearance: jnp.ndarray


@dataclass(frozen=True)
class ProjectionProblem:
    physical_desired: jnp.ndarray
    alpha_desired: jnp.ndarray
    z_desired: jnp.ndarray
    a_matrix: jnp.ndarray
    b_vector: jnp.ndarray
    inverse_weight: jnp.ndarray
    lower_bound: jnp.ndarray
    upper_bound: jnp.ndarray


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
        self.key = jax.random.PRNGKey(seed)
        self.num_barriers = len(obstacle_field.obstacles)
        self.augmented_control_dim = robot.control_dim + self.num_barriers if algo == "brmppi" else robot.control_dim
        self.control_sequence = jnp.zeros((self.config.horizon, self.augmented_control_dim), dtype=float)
        self.alpha_state = jnp.full((self.num_barriers,), self.config.barrier_alpha, dtype=float)
        self.last_diagnostics: dict[str, object] = {}

    @property
    def barrier_source(self) -> str:
        if hasattr(self.sdf_model, "obstacle_barriers"):
            return "neural_sdf"
        return "analytic_sdf"

    def command(self, state: jnp.ndarray, goal: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, object]]:
        result = self._command_jit(
            jnp.asarray(state, dtype=float),
            jnp.asarray(goal, dtype=float),
            self.control_sequence,
            self.alpha_state,
            self.key,
        )
        (
            action,
            updated_sequence,
            alpha_next,
            key_next,
            costs,
            min_clearances,
            trajectories,
            effective_physical,
            effective_augmented,
            alpha_traces,
            weights,
            projected_first,
        ) = jax.block_until_ready(result)

        self.control_sequence = updated_sequence
        self.alpha_state = alpha_next
        self.key = key_next

        best_idx = int(jax.device_get(jnp.argmin(costs)))
        sampled_collision_count = int(jax.device_get(jnp.count_nonzero(min_clearances < 0.0)))
        plot_count = min(self.config.plot_samples, self.config.samples)
        weighted_alpha_terminal = (
            jnp.einsum("s,sn->n", weights, alpha_traces[:, -1, :])
            if self.algo == "brmppi"
            else jnp.array([], dtype=float)
        )
        diagnostics: dict[str, object] = {
            "barrier_source": self.barrier_source,
            "best_cost": float(jax.device_get(costs[best_idx])),
            "mean_cost": float(jax.device_get(jnp.mean(costs))),
            "best_min_clearance": float(jax.device_get(min_clearances[best_idx])),
            "best_collision": bool(jax.device_get(min_clearances[best_idx] < 0.0)),
            "sampled_min_clearance": float(jax.device_get(jnp.min(min_clearances))),
            "sampled_collision_count": sampled_collision_count,
            "sampled_collision_fraction": float(sampled_collision_count / self.config.samples),
            "weighted_min_clearance": float(jax.device_get(jnp.sum(weights * min_clearances))),
            "sampled_costs": costs.copy(),
            "sampled_min_clearances": min_clearances.copy(),
            "best_trajectory": trajectories[best_idx],
            "best_controls": effective_physical[best_idx],
            "best_augmented_controls": effective_augmented[best_idx],
            "best_alpha_trace": alpha_traces[best_idx],
            "weighted_alpha_terminal": weighted_alpha_terminal,
            "projected_first_augmented": projected_first,
            "sampled_trajectories": trajectories[:plot_count],
        }
        self.last_diagnostics = diagnostics
        return action, diagnostics

    @partial(jax.jit, static_argnums=(0,))
    def _command_jit(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        control_sequence: jnp.ndarray,
        alpha_state: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, ...]:
        nominal = self._nominal_augmented_control_jax(state, goal)
        nominal_sequence = jnp.tile(nominal[None, :], (self.config.horizon, 1))
        sequence_is_fresh = jnp.logical_not(jnp.any(control_sequence != 0.0))
        base_sequence = jnp.where(sequence_is_fresh, nominal_sequence, control_sequence)

        candidates, key_next = self._sample_control_sequences_jax(base_sequence, key)
        rollout_fn = lambda controls: self._rollout_jax(state, goal, controls, alpha_state)
        (
            costs,
            trajectories,
            effective_physical,
            effective_augmented,
            alpha_traces,
            min_clearances,
        ) = jax.vmap(rollout_fn)(candidates)

        weights = self._weights_jax(costs)
        if self.algo == "brmppi":
            updated = jnp.einsum("s,shm->hm", weights, candidates)
            updated = self._clip_augmented_controls_jax(updated)
            action, alpha_next, projected_first = self._project_augmented_control_jax(state, updated[0], alpha_state)
        else:
            updated = jnp.einsum("s,shm->hm", weights, effective_physical)
            updated = self.robot.clip_controls(updated)
            action = updated[0]
            alpha_next = alpha_state
            projected_first = updated[0]

        updated_sequence = jnp.concatenate((updated[1:], updated[-1:]), axis=0)
        return (
            action,
            updated_sequence,
            alpha_next,
            key_next,
            costs,
            min_clearances,
            trajectories,
            effective_physical,
            effective_augmented,
            alpha_traces,
            weights,
            projected_first,
        )

    def _nominal_augmented_control(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        return self._nominal_augmented_control_jax(jnp.asarray(state, dtype=float), jnp.asarray(goal, dtype=float))

    def _nominal_augmented_control_jax(self, state: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
        physical = self.robot.nominal_control(state, goal)
        if self.algo != "brmppi":
            return physical
        alpha_rate = -self.config.alpha_rate_bound * jnp.ones(self.num_barriers, dtype=float)
        return jnp.concatenate((physical, alpha_rate))

    def _sample_control_sequences_jax(self, base_sequence: jnp.ndarray, key: jax.Array) -> tuple[jnp.ndarray, jax.Array]:
        cfg = self.config
        key_next, noise_key = jax.random.split(key)
        if self.algo == "brmppi":
            bounds = self._augmented_control_bounds_jax()
            physical_sigma = cfg.noise_scale * (self.robot.control_bounds[:, 1] - self.robot.control_bounds[:, 0])
            alpha_sigma = cfg.alpha_noise_scale * (bounds[self.robot.control_dim :, 1] - bounds[self.robot.control_dim :, 0])
            sigma = jnp.concatenate((physical_sigma, alpha_sigma))
        else:
            bounds = self.robot.control_bounds
            sigma = cfg.noise_scale * (bounds[:, 1] - bounds[:, 0])

        noise = jax.random.normal(
            noise_key,
            shape=(cfg.samples, cfg.horizon, self.augmented_control_dim),
            dtype=base_sequence.dtype,
        ) * sigma
        candidates = base_sequence[None, :, :] + noise
        candidates = candidates.at[0].set(base_sequence)
        if self.algo == "brmppi":
            return self._clip_augmented_controls_jax(candidates), key_next
        return self.robot.clip_controls(candidates), key_next

    def _weights(self, costs: jnp.ndarray) -> jnp.ndarray:
        return self._weights_jax(jnp.asarray(costs, dtype=float))

    def _weights_jax(self, costs: jnp.ndarray) -> jnp.ndarray:
        shifted = costs - jnp.min(costs)
        scaled = -shifted / max(self.config.temperature, 1e-6)
        scaled = jnp.clip(scaled, -60.0, 0.0)
        weights = jnp.exp(scaled)
        total = jnp.sum(weights)
        fallback = jnp.ones_like(costs) / costs.size
        normalized = weights / total
        return jnp.where(jnp.isfinite(total) & (total > 0.0), normalized, fallback)

    def _rollout(self, state: jnp.ndarray, goal: jnp.ndarray, controls: jnp.ndarray) -> RolloutResult:
        cost, states, effective_controls, effective_augmented, alpha_trace, min_clearance = self._rollout_jax(
            jnp.asarray(state, dtype=float),
            jnp.asarray(goal, dtype=float),
            jnp.asarray(controls, dtype=float),
            self.alpha_state,
        )
        return RolloutResult(cost, states, effective_controls, effective_augmented, alpha_trace, min_clearance)

    def _rollout_jax(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        controls: jnp.ndarray,
        initial_alpha: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        initial_clearance = self._clearance_jax(state)
        initial_cost = jnp.array(0.0, dtype=float)
        initial_carry = (state, initial_alpha, initial_cost, initial_clearance)

        def step_scan(carry, desired_control):
            x, alpha, cost, min_clearance = carry
            if self.algo == "brmppi":
                u, alpha_next, projected_augmented = self._project_augmented_control_jax(x, desired_control, alpha)
            else:
                u = desired_control
                alpha_next = alpha
                projected_augmented = desired_control

            x_next = self.robot.step(x, u, cfg.dt)
            h_next = self._clearance_jax(x_next)
            goal_error = jnp.linalg.norm(self.robot.position(x_next) - goal)
            running_cost = cfg.goal_weight * goal_error**2
            running_cost += cfg.control_weight * jnp.dot(u, u)

            if self.algo == "penalty_mppi":
                running_cost += self._safety_cost_jax(h_next)
                running_cost += jnp.where(
                    h_next < 0.0,
                    cfg.collision_weight * (1.0 + jnp.abs(h_next)) ** 2,
                    0.0,
                )
            elif self.algo == "brmppi":
                running_cost += self._barrier_alpha_cost_jax(x, alpha_next)
                running_cost += self._br_safety_cost_jax(h_next)

            next_carry = (x_next, alpha_next, cost + running_cost, jnp.minimum(min_clearance, h_next))
            return next_carry, (x_next, u, projected_augmented, alpha_next)

        (x_final, _alpha_final, running_cost, min_clearance), outputs = jax.lax.scan(
            step_scan,
            initial_carry,
            controls,
        )
        states_next, effective_controls, effective_augmented, alpha_nexts = outputs
        states = jnp.concatenate((state[None, :], states_next), axis=0)
        if self.algo == "brmppi":
            alpha_trace = jnp.concatenate((initial_alpha[None, :], alpha_nexts), axis=0)
        else:
            alpha_trace = jnp.zeros((cfg.horizon + 1, self.num_barriers), dtype=float)
        final_error = jnp.linalg.norm(self.robot.position(x_final) - goal)
        total_cost = running_cost + cfg.final_goal_weight * final_error**2
        return total_cost, states, effective_controls, effective_augmented, alpha_trace, min_clearance

    def _rollout_safety_diagnostics(
        self,
        min_clearances: jnp.ndarray,
        weights: jnp.ndarray,
        best_idx: int,
    ) -> dict[str, float | bool | int]:
        sampled_collision_count = int(jax.device_get(jnp.count_nonzero(min_clearances < 0.0)))
        return {
            "best_min_clearance": float(jax.device_get(min_clearances[best_idx])),
            "best_collision": bool(jax.device_get(min_clearances[best_idx] < 0.0)),
            "sampled_min_clearance": float(jax.device_get(jnp.min(min_clearances))),
            "sampled_collision_count": sampled_collision_count,
            "sampled_collision_fraction": float(sampled_collision_count / self.config.samples),
            "weighted_min_clearance": float(jax.device_get(jnp.sum(weights * min_clearances))),
        }

    def _safety_cost(self, clearance: float) -> float:
        return float(jax.device_get(self._safety_cost_jax(jnp.asarray(clearance, dtype=float))))

    def _safety_cost_jax(self, clearance: jnp.ndarray) -> jnp.ndarray:
        margin = 0.75
        cost = self.config.safety_weight * (margin - clearance) ** 2
        collision_cost = cost + self.config.collision_weight
        return jnp.where(clearance >= margin, 0.0, jnp.where(clearance <= 0.0, collision_cost, cost))

    def _br_safety_cost(self, clearance: float) -> float:
        return float(jax.device_get(self._br_safety_cost_jax(jnp.asarray(clearance, dtype=float))))

    def _br_safety_cost_jax(self, clearance: jnp.ndarray) -> jnp.ndarray:
        margin = self.config.br_clearance_margin
        cost = self.config.br_clearance_weight * (margin - clearance) ** 2
        cost = cost + jnp.where(
            clearance <= 0.0,
            self.config.br_collision_weight * (1.0 + jnp.abs(clearance)) ** 2,
            0.0,
        )
        return jnp.where(clearance >= margin, 0.0, cost)

    def _clearance(self, state: jnp.ndarray) -> float:
        return float(jax.device_get(self._clearance_jax(jnp.asarray(state, dtype=float))))

    def _clearance_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        points = self.robot.body_points(state)
        distances = self.obstacle_field.signed_distance(points)
        return jnp.min(distances) - self.robot.body_point_radius

    def _barrier_values(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._barrier_values_jax(jnp.asarray(state, dtype=float))

    def _barrier_values_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        if hasattr(self.sdf_model, "obstacle_barriers"):
            return jnp.asarray(self.sdf_model.obstacle_barriers(state, self.obstacle_field), dtype=float)
        return self._analytic_obstacle_barriers_jax(state)

    def _analytic_obstacle_barriers(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._analytic_obstacle_barriers_jax(jnp.asarray(state, dtype=float))

    def _analytic_obstacle_barriers_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        points = self.robot.body_points(state)
        centers = self.obstacle_field.centers
        radii = self.obstacle_field.radii
        distances = jnp.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2) - radii[None, :]
        return jnp.min(distances, axis=0) - self.robot.body_point_radius

    def _barrier_alpha_cost(self, state: jnp.ndarray, alpha: jnp.ndarray) -> float:
        return float(
            jax.device_get(
                self._barrier_alpha_cost_jax(jnp.asarray(state, dtype=float), jnp.asarray(alpha, dtype=float))
            )
        )

    def _barrier_alpha_cost_jax(self, state: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.array(0.0, dtype=float)
        cfg = self.config
        h = self._projection_barrier_values_jax(state)
        hmin_idx = jnp.argmin(h)
        hmin = h[hmin_idx]
        alpha_hmin = alpha[hmin_idx]
        cost = cfg.barrier_alpha_cost_weight * alpha_hmin / jnp.maximum(hmin, 0.01)
        return jnp.where((hmin <= 0.0) | (hmin >= cfg.barrier_buffer_distance), 0.0, cost)

    def _project_augmented_control(
        self,
        state: jnp.ndarray,
        augmented_control: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._project_augmented_control_jax(
            jnp.asarray(state, dtype=float),
            jnp.asarray(augmented_control, dtype=float),
            jnp.asarray(alpha, dtype=float),
        )

    def _project_augmented_control_jax(
        self,
        state: jnp.ndarray,
        augmented_control: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        problem = self._projection_problem_jax(state, augmented_control, alpha)
        if self.num_barriers == 0:
            return problem.physical_desired, problem.alpha_desired, problem.z_desired

        projected = self._closed_form_projection_jax(
            problem.z_desired,
            problem.a_matrix,
            problem.b_vector,
            problem.inverse_weight,
            problem.lower_bound,
            problem.upper_bound,
        )
        projected = self._repair_projected_control_bounds_jax(projected, problem.a_matrix, problem.b_vector)
        return projected[: self.robot.control_dim], projected[self.robot.control_dim :], projected

    def _projection_problem(
        self,
        state: jnp.ndarray,
        augmented_control: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> ProjectionProblem:
        return self._projection_problem_jax(
            jnp.asarray(state, dtype=float),
            jnp.asarray(augmented_control, dtype=float),
            jnp.asarray(alpha, dtype=float),
        )

    def _projection_problem_jax(
        self,
        state: jnp.ndarray,
        augmented_control: jnp.ndarray,
        alpha: jnp.ndarray,
    ) -> ProjectionProblem:
        cfg = self.config
        physical_desired = self.robot.clip_control(augmented_control[: self.robot.control_dim])
        alpha_rate = jnp.clip(
            augmented_control[self.robot.control_dim :],
            -cfg.alpha_rate_bound,
            cfg.alpha_rate_bound,
        )
        alpha_desired = jnp.clip(
            alpha + alpha_rate * cfg.dt,
            -cfg.alpha_state_bound,
            cfg.alpha_state_bound,
        )
        z_desired = jnp.concatenate((physical_desired, alpha_desired))
        lower, upper = self._projection_bounds_jax()
        inverse_weight = jnp.concatenate(
            (
                jnp.ones(self.robot.control_dim, dtype=float),
                cfg.alpha_projection_inverse_weight * jnp.ones(self.num_barriers, dtype=float),
            )
        )
        if self.num_barriers == 0:
            return ProjectionProblem(
                physical_desired=physical_desired,
                alpha_desired=alpha_desired,
                z_desired=z_desired,
                a_matrix=jnp.zeros((0, self.augmented_control_dim), dtype=float),
                b_vector=jnp.zeros(0, dtype=float),
                inverse_weight=inverse_weight,
                lower_bound=lower,
                upper_bound=upper,
            )

        h = self._projection_barrier_values_with_margin_jax(state)
        barrier_jacobian = self._projection_barrier_jacobian_jax(state)
        drift = self.robot.drift(state)
        control_matrix = self.robot.control_matrix(state)
        physical_block = barrier_jacobian @ control_matrix
        alpha_block = jnp.diag(h)
        a_matrix = jnp.concatenate((physical_block, alpha_block), axis=1)
        b_vector = -(barrier_jacobian @ drift)
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

    def _projection_barrier_jacobian(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_jacobian_jax(jnp.asarray(state, dtype=float))

    def _projection_barrier_jacobian_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        h_base = self._projection_barrier_values_with_margin_jax(state)
        eps = 1e-4
        perturbations = eps * jnp.eye(self.robot.state_dim, dtype=float)
        diff_fn = lambda delta: (self._projection_barrier_values_with_margin_jax(state + delta) - h_base) / eps
        return jax.vmap(diff_fn)(perturbations).T

    def _projection_barrier_values(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_values_jax(jnp.asarray(state, dtype=float))

    def _projection_barrier_values_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        barrier_state = self.robot.projection_barrier_state(state, self.config.dt)
        return self._barrier_values_jax(barrier_state)

    def _projection_barrier_values_with_margin(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_values_with_margin_jax(jnp.asarray(state, dtype=float))

    def _projection_barrier_values_with_margin_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_values_jax(state) - self.config.barrier_projection_margin

    def _closed_form_projection(
        self,
        z_des: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
        inverse_weight_diag: jnp.ndarray,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._closed_form_projection_jax(
            jnp.asarray(z_des, dtype=float),
            jnp.asarray(a_matrix, dtype=float),
            jnp.asarray(b_vector, dtype=float),
            jnp.asarray(inverse_weight_diag, dtype=float),
            jnp.asarray(lower, dtype=float),
            jnp.asarray(upper, dtype=float),
        )

    def _closed_form_projection_jax(
        self,
        z_des: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
        inverse_weight_diag: jnp.ndarray,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
    ) -> jnp.ndarray:
        rho = max(0.0, self.config.bound_penalty)
        if rho > 0.0:
            weight_diag = 1.0 / jnp.maximum(inverse_weight_diag, 1e-9)
            m_diag = weight_diag + rho
            m_inv_diag = 1.0 / m_diag
            p = weight_diag * z_des + 0.5 * rho * (lower + upper)
            am_inv = a_matrix * m_inv_diag[None, :]
            rhs = am_inv @ p - b_vector
            correction = self._solve_projection_system_jax(a_matrix, rhs, m_inv_diag)
            return m_inv_diag * (p - a_matrix.T @ correction)

        return self._unbounded_closed_form_projection_jax(z_des, a_matrix, b_vector, inverse_weight_diag)

    def _unbounded_closed_form_projection(
        self,
        z_des: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
        inverse_weight_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._unbounded_closed_form_projection_jax(
            jnp.asarray(z_des, dtype=float),
            jnp.asarray(a_matrix, dtype=float),
            jnp.asarray(b_vector, dtype=float),
            jnp.asarray(inverse_weight_diag, dtype=float),
        )

    def _unbounded_closed_form_projection_jax(
        self,
        z_des: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
        inverse_weight_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        weighted_a_t = inverse_weight_diag[:, None] * a_matrix.T
        rhs = b_vector - a_matrix @ z_des
        lagrange = self._solve_projection_system_jax(a_matrix, rhs, inverse_weight_diag)
        return z_des + weighted_a_t @ lagrange

    def _solve_projection_system(
        self,
        a_matrix: jnp.ndarray,
        rhs: jnp.ndarray,
        inverse_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._solve_projection_system_jax(
            jnp.asarray(a_matrix, dtype=float),
            jnp.asarray(rhs, dtype=float),
            jnp.asarray(inverse_diag, dtype=float),
        )

    def _solve_projection_system_jax(
        self,
        a_matrix: jnp.ndarray,
        rhs: jnp.ndarray,
        inverse_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        if self._has_single_alpha_per_row_static():
            return self._solve_diagonal_plus_low_rank_jax(a_matrix, rhs, inverse_diag)

        lhs = a_matrix @ (inverse_diag[:, None] * a_matrix.T)
        return self._solve_with_pinv_fallback(lhs + 1e-8 * jnp.eye(lhs.shape[0], dtype=float), rhs, lhs)

    def _has_single_alpha_per_row(self, a_matrix: jnp.ndarray) -> bool:
        return self._has_single_alpha_per_row_static()

    def _has_single_alpha_per_row_static(self) -> bool:
        return self.algo == "brmppi" and self.num_barriers > 0

    def _solve_diagonal_plus_low_rank(
        self,
        a_matrix: jnp.ndarray,
        rhs: jnp.ndarray,
        inverse_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._solve_diagonal_plus_low_rank_jax(
            jnp.asarray(a_matrix, dtype=float),
            jnp.asarray(rhs, dtype=float),
            jnp.asarray(inverse_diag, dtype=float),
        )

    def _solve_diagonal_plus_low_rank_jax(
        self,
        a_matrix: jnp.ndarray,
        rhs: jnp.ndarray,
        inverse_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        control_dim = self.robot.control_dim
        au = a_matrix[:, :control_dim]
        alpha_block = a_matrix[:, control_dim:]
        inv_u = jnp.maximum(inverse_diag[:control_dim], 1e-12)
        inv_alpha = jnp.maximum(inverse_diag[control_dim:], 1e-12)
        diag_terms = (alpha_block**2) @ inv_alpha + 1e-8
        diag_inv = 1.0 / diag_terms
        diag_inv_au = diag_inv[:, None] * au
        small_lhs = jnp.diag(1.0 / inv_u) + au.T @ diag_inv_au
        small_rhs = au.T @ (diag_inv * rhs)
        small_solution = self._solve_with_pinv_fallback(small_lhs, small_rhs, small_lhs)
        return diag_inv * rhs - diag_inv_au @ small_solution

    def _solve_with_pinv_fallback(self, solve_lhs: jnp.ndarray, rhs: jnp.ndarray, pinv_lhs: jnp.ndarray) -> jnp.ndarray:
        solution = jnp.linalg.solve(solve_lhs, rhs)
        fallback = jnp.linalg.pinv(pinv_lhs) @ rhs
        return jnp.where(jnp.all(jnp.isfinite(solution)), solution, fallback)

    def _repair_projected_control_bounds(
        self,
        projected: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        return self._repair_projected_control_bounds_jax(
            jnp.asarray(projected, dtype=float),
            jnp.asarray(a_matrix, dtype=float),
            jnp.asarray(b_vector, dtype=float),
        )

    def _repair_projected_control_bounds_jax(
        self,
        projected: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        control_dim = self.robot.control_dim
        control_lower = self.robot.control_bounds[:, 0]
        control_upper = self.robot.control_bounds[:, 1]
        bounded_control = jnp.clip(projected[:control_dim], control_lower, control_upper)
        changed = jnp.any(jnp.abs(bounded_control - projected[:control_dim]) > 1e-10)

        alpha_block = a_matrix[:, control_dim:]
        alpha_rhs = b_vector - a_matrix[:, :control_dim] @ bounded_control
        alpha_diag = jnp.diag(alpha_block)
        repaired_alpha = jnp.where(
            jnp.abs(alpha_diag) > 1e-8,
            alpha_rhs / alpha_diag,
            projected[control_dim:],
        )
        repaired = jnp.concatenate((bounded_control, repaired_alpha))
        return jnp.where(changed, repaired, projected)

    def _augmented_control_bounds(self) -> jnp.ndarray:
        return self._augmented_control_bounds_jax()

    def _augmented_control_bounds_jax(self) -> jnp.ndarray:
        alpha_bounds = jnp.column_stack(
            (
                -self.config.alpha_rate_bound * jnp.ones(self.num_barriers, dtype=float),
                self.config.alpha_rate_bound * jnp.ones(self.num_barriers, dtype=float),
            )
        )
        return jnp.vstack((self.robot.control_bounds, alpha_bounds))

    def _projection_bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._projection_bounds_jax()

    def _projection_bounds_jax(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        lower = jnp.concatenate(
            (
                self.robot.control_bounds[:, 0],
                -self.config.alpha_state_bound * jnp.ones(self.num_barriers, dtype=float),
            )
        )
        upper = jnp.concatenate(
            (
                self.robot.control_bounds[:, 1],
                self.config.alpha_state_bound * jnp.ones(self.num_barriers, dtype=float),
            )
        )
        return lower, upper

    def _clip_augmented_controls(self, controls: jnp.ndarray) -> jnp.ndarray:
        return self._clip_augmented_controls_jax(jnp.asarray(controls, dtype=float))

    def _clip_augmented_controls_jax(self, controls: jnp.ndarray) -> jnp.ndarray:
        bounds = self._augmented_control_bounds_jax()
        return jnp.clip(controls, bounds[:, 0], bounds[:, 1])
