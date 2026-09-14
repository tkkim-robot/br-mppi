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


ALGORITHMS = (
    "brmppi",
    "mppi",
    "penalty_mppi",
    "mppi_cbf",
    "shield_mppi",
    "sc_mppi",
    "gs_mppi",
)


class NeuralBarrierModel(Protocol):
    def obstacle_barriers(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> jnp.ndarray:
        ...

    def obstacle_barriers_and_jacobian(
        self,
        state: jnp.ndarray,
        field: ObstacleField,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    cbf_alpha: float = 1.0
    cbf_qp_iterations: int = 35
    cbf_qp_rho: float = 8.0
    cbf_qp_regularization: float = 1e-8
    shield_cbf_penalty_weight: float = 900.0
    shield_alpha: float = 0.98
    shield_repair_horizon: int = 6
    shield_repair_steps: int = 8
    shield_repair_step_size: float = 0.04
    sc_barrier_eps: float = 1e-3
    sc_barrier_gamma: float = 0.25
    sc_feedback_gain: float = 0.08
    sc_feedback_clip: float = 0.7
    sc_feedback_iterations: int = 1
    sc_feedback_regularization: float = 1e-4
    sc_beta_cost_weight: float = 8.0
    sc_state_cost_weight: float = 0.05
    sc_control_cost_weight: float = 0.08
    sc_use_combined_barrier: bool = True
    gs_softmin_rho: float = 25.0
    gs_closed_form_gamma: float = 2.0
    gs_alpha: float = 1.0
    gs_composite_margin: float = 0.0
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
        sdf_model: NeuralBarrierModel | None = None,
        config: MPPIConfig | None = None,
        seed: int = 7,
    ) -> None:
        if algo not in ALGORITHMS:
            raise ValueError(f"Unknown MPPI algorithm '{algo}'. Choose one of {ALGORITHMS}.")
        if sdf_model is not None and algo != "brmppi":
            raise NotImplementedError(
                f"A neural SDF is only implemented for brmppi, not {algo}."
            )
        if sdf_model is not None:
            required_methods = ("obstacle_barriers", "obstacle_barriers_and_jacobian")
            missing_methods = [
                method_name
                for method_name in required_methods
                if not callable(getattr(sdf_model, method_name, None))
            ]
            if missing_methods:
                raise TypeError(
                    "sdf_model must implement the optimized neural barrier API; missing: "
                    + ", ".join(missing_methods)
                )
        self.robot = robot
        self.obstacle_field = obstacle_field
        self.sdf_model = sdf_model
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
        return "neural_sdf" if self.sdf_model is not None else "analytic_sdf"

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
            shield_repair_violation_before,
            shield_repair_violation_after,
            sc_feedback_gain_norm,
            gs_cbf_residual_after_clipping,
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
        if self.algo == "brmppi" and self.sdf_model is not None:
            # BR-MPPI does not use the CBF-QP residual, so skip its neural evaluation.
            constraint_residual = None
        else:
            constraint_residual = self._cbf_constraint_residual_jax(
                jnp.asarray(state, dtype=float),
                action,
            )
        next_action_state = self.robot.step(jnp.asarray(state, dtype=float), action, self.config.dt)
        shield_penalty = (
            self._shield_dcbf_penalty_jax(jnp.asarray(state, dtype=float), next_action_state)
            if self.algo == "shield_mppi"
            else jnp.array(0.0, dtype=float)
        )
        gs_composite_barrier = (
            self._gs_composite_barrier_jax(jnp.asarray(state, dtype=float))
            if self.algo == "gs_mppi"
            else jnp.array(0.0, dtype=float)
        )
        diagnostics: dict[str, object] = {
            "barrier_source": self.barrier_source,
            "uses_cbf_qp": self._uses_cbf_qp_static(),
            "uses_rollout_cbf_qp": self._uses_rollout_cbf_qp_static(),
            "cbf_constraint_max_violation": (
                None
                if constraint_residual is None
                else float(jax.device_get(constraint_residual))
            ),
            "cbf_qp_max_violation": (
                float(jax.device_get(constraint_residual))
                if self._uses_cbf_qp_static()
                else 0.0
            ),
            "shield_dcbf_penalty": float(jax.device_get(shield_penalty)),
            "shield_repair_violation_before": float(jax.device_get(shield_repair_violation_before)),
            "shield_repair_violation_after": float(jax.device_get(shield_repair_violation_after)),
            "uses_shield_repair": self.algo == "shield_mppi",
            "sc_feedback_norm": float(jax.device_get(sc_feedback_gain_norm)),
            "sc_feedback_model": "dbas_ilqr_feedback" if self.algo == "sc_mppi" else "none",
            "gs_composite_barrier": float(jax.device_get(gs_composite_barrier)),
            "gs_cbf_residual_after_clipping": float(jax.device_get(gs_cbf_residual_after_clipping)),
            "uses_gs_closed_form_safe_control": self.algo == "gs_mppi",
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

        sc_xbar_ref = jnp.zeros((self.config.horizon + 1, self._sc_xbar_dim_static()), dtype=float)
        sc_feedback_gains = jnp.zeros(
            (self.config.horizon, self.robot.control_dim, self._sc_xbar_dim_static()),
            dtype=float,
        )
        sampling_base_sequence = base_sequence
        if self.algo == "sc_mppi":
            sampling_base_sequence, sc_xbar_ref, sc_feedback_gains = self._compute_safe_feedback_jax(
                state,
                base_sequence,
            )

        candidates, key_next = self._sample_control_sequences_jax(sampling_base_sequence, key)
        rollout_fn = lambda controls: self._rollout_jax(
            state,
            goal,
            controls,
            alpha_state,
            sc_xbar_ref,
            sc_feedback_gains,
        )
        (
            costs,
            trajectories,
            effective_physical,
            effective_augmented,
            alpha_traces,
            min_clearances,
        ) = jax.vmap(rollout_fn)(candidates)

        weights = self._weights_jax(costs)
        best_idx = jnp.argmin(costs)
        shield_repair_violation_before = jnp.array(0.0, dtype=float)
        shield_repair_violation_after = jnp.array(0.0, dtype=float)
        sc_feedback_gain_norm = jnp.array(0.0, dtype=float)
        gs_cbf_residual_after_clipping = jnp.array(0.0, dtype=float)
        if self.algo == "brmppi":
            updated = jnp.einsum("s,shm->hm", weights, candidates)
            updated = self._clip_augmented_controls_jax(updated)
            action, alpha_next, projected_first = self._project_augmented_control_jax(state, updated[0], alpha_state)
        elif self.algo == "mppi_cbf":
            updated = jnp.einsum("s,shm->hm", weights, effective_physical)
            updated = self.robot.clip_controls(updated)
            action = self._cbf_qp_filter_jax(state, updated[0])
            alpha_next = alpha_state
            projected_first = action
        elif self.algo == "shield_mppi":
            updated = jnp.einsum("s,shm->hm", weights, candidates)
            updated = self.robot.clip_controls(updated)
            shield_repair_violation_before = self._shield_repair_violation_jax(state, updated)
            repaired = self._shield_repair_sequence_jax(state, updated)
            shield_repair_violation_after = self._shield_repair_violation_jax(state, repaired)
            action = repaired[0]
            alpha_next = alpha_state
            projected_first = action
        elif self.algo == "gs_mppi":
            updated = jnp.einsum("s,shm->hm", weights, candidates)
            updated = self.robot.clip_controls(updated)
            action = self._gs_safe_control_jax(state, candidates[best_idx, 0])
            gs_cbf_residual_after_clipping = self._gs_cbf_residual_jax(state, action)
            alpha_next = alpha_state
            projected_first = action
        elif self.algo == "sc_mppi":
            updated = jnp.einsum("s,shm->hm", weights, candidates)
            updated = self.robot.clip_controls(updated)
            updated, updated_xbar_ref, updated_feedback_gains = self._compute_safe_feedback_jax(state, updated)
            sc_feedback_gain_norm = jnp.linalg.norm(updated_feedback_gains[0])
            xbar = self._sc_initial_xbar_jax(state)
            feedback = updated_feedback_gains[0] @ (xbar - updated_xbar_ref[0])
            action = self.robot.clip_control(updated[0] + feedback)
            alpha_next = alpha_state
            projected_first = action
        else:
            updated = jnp.einsum("s,shm->hm", weights, effective_physical)
            updated = self.robot.clip_controls(updated)
            action = updated[0]
            alpha_next = alpha_state
            projected_first = action

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
            shield_repair_violation_before,
            shield_repair_violation_after,
            sc_feedback_gain_norm,
            gs_cbf_residual_after_clipping,
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
        state_jax = jnp.asarray(state, dtype=float)
        controls_jax = jnp.asarray(controls, dtype=float)
        sc_xbar_ref = jnp.zeros((self.config.horizon + 1, self._sc_xbar_dim_static()), dtype=float)
        sc_feedback_gains = jnp.zeros(
            (self.config.horizon, self.robot.control_dim, self._sc_xbar_dim_static()),
            dtype=float,
        )
        if self.algo == "sc_mppi":
            _u_safe, sc_xbar_ref, sc_feedback_gains = self._compute_safe_feedback_jax(state_jax, controls_jax)
        cost, states, effective_controls, effective_augmented, alpha_trace, min_clearance = self._rollout_jax(
            state_jax,
            jnp.asarray(goal, dtype=float),
            controls_jax,
            self.alpha_state,
            sc_xbar_ref,
            sc_feedback_gains,
        )
        return RolloutResult(cost, states, effective_controls, effective_augmented, alpha_trace, min_clearance)

    def _rollout_jax(
        self,
        state: jnp.ndarray,
        goal: jnp.ndarray,
        controls: jnp.ndarray,
        initial_alpha: jnp.ndarray,
        sc_xbar_ref: jnp.ndarray,
        sc_feedback_gains: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        initial_clearance = self._clearance_jax(state)
        initial_sc_beta = self._sc_initial_beta_jax(state)
        initial_cost = jnp.array(0.0, dtype=float)
        initial_carry = (state, initial_alpha, initial_sc_beta, initial_cost, initial_clearance)

        def step_scan(carry, step_inputs):
            step_idx, desired_control = step_inputs
            x, alpha, sc_beta, cost, min_clearance = carry
            if self.algo == "brmppi":
                u, alpha_next, projected_augmented = self._project_augmented_control_jax(x, desired_control, alpha)
            elif self.algo == "sc_mppi":
                xbar = self._sc_xbar_jax(x, sc_beta)
                feedback = sc_feedback_gains[step_idx] @ (xbar - sc_xbar_ref[step_idx])
                feedback = jnp.clip(feedback, -self.config.sc_feedback_clip, self.config.sc_feedback_clip)
                u = self.robot.clip_control(desired_control + feedback)
                alpha_next = alpha
                projected_augmented = desired_control
            elif self.algo == "gs_mppi":
                u = self._gs_safe_control_jax(x, desired_control)
                alpha_next = alpha
                projected_augmented = desired_control
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
            elif self.algo == "shield_mppi":
                running_cost += self._shield_dcbf_penalty_jax(x, x_next)

            sc_beta_next = (
                self._sc_next_beta_jax(sc_beta, x, x_next)
                if self.algo == "sc_mppi"
                else sc_beta
            )
            next_carry = (x_next, alpha_next, sc_beta_next, cost + running_cost, jnp.minimum(min_clearance, h_next))
            return next_carry, (x_next, u, projected_augmented, alpha_next)

        (x_final, _alpha_final, _sc_beta_final, running_cost, min_clearance), outputs = jax.lax.scan(
            step_scan,
            initial_carry,
            (jnp.arange(self.config.horizon), controls),
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

    def _shield_dcbf_penalty_jax(self, state: jnp.ndarray, next_state: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.array(0.0, dtype=float)
        h_prev = self._projection_barrier_values_with_margin_jax(state)
        h_next = self._projection_barrier_values_with_margin_jax(next_state)
        violation = jnp.maximum(self.config.shield_alpha * h_prev - h_next, 0.0)
        return self.config.shield_cbf_penalty_weight * jnp.sum(violation)

    def _shield_repair_violation_jax(self, state: jnp.ndarray, sequence: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.array(0.0, dtype=float)
        repair_horizon = min(self.config.shield_repair_horizon, self.config.horizon)

        def step_scan(carry, control):
            x, total = carry
            x_next = self.robot.step(x, control, self.config.dt)
            h_prev = self._projection_barrier_values_with_margin_jax(x)
            h_next = self._projection_barrier_values_with_margin_jax(x_next)
            violation = jnp.maximum(self.config.shield_alpha * h_prev - h_next, 0.0)
            return (x_next, total + jnp.sum(violation)), None

        (_x_final, total_violation), _unused = jax.lax.scan(
            step_scan,
            (state, jnp.array(0.0, dtype=float)),
            self.robot.clip_controls(sequence[:repair_horizon]),
        )
        return total_violation

    def _shield_repair_sequence_jax(self, state: jnp.ndarray, sequence: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0 or self.config.shield_repair_steps <= 0:
            return self.robot.clip_controls(sequence)

        repair_horizon = min(self.config.shield_repair_horizon, self.config.horizon)
        prefix = sequence[:repair_horizon]
        suffix = sequence[repair_horizon:]

        def repair_objective(prefix_controls: jnp.ndarray) -> jnp.ndarray:
            def step_scan(carry, control):
                x, objective = carry
                x_next = self.robot.step(x, control, self.config.dt)
                h_prev = self._projection_barrier_values_with_margin_jax(x)
                h_next = self._projection_barrier_values_with_margin_jax(x_next)
                violation = self.config.shield_alpha * h_prev - h_next
                # Squared ReLU is a JAX-friendly local surrogate for the paper's min/max DCBF violation.
                smooth_violation = jnp.maximum(violation, 0.0)
                objective_next = objective + jnp.sum(smooth_violation**2)
                return (x_next, objective_next), None

            (_x_final, objective), _unused = jax.lax.scan(
                step_scan,
                (state, jnp.array(0.0, dtype=float)),
                prefix_controls,
            )
            return objective

        def repair_step(prefix_controls, _unused):
            gradient = jax.grad(repair_objective)(prefix_controls)
            repaired = prefix_controls - self.config.shield_repair_step_size * gradient
            return self.robot.clip_controls(repaired), None

        repaired_prefix, _unused = jax.lax.scan(
            repair_step,
            self.robot.clip_controls(prefix),
            jnp.arange(self.config.shield_repair_steps),
        )
        return jnp.concatenate((repaired_prefix, suffix), axis=0)

    def _sc_initial_beta_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        h = self._sc_barrier_values_jax(state)
        return self._sc_inverse_barrier_jax(h)

    def _sc_next_beta_jax(self, beta: jnp.ndarray, state: jnp.ndarray, next_state: jnp.ndarray) -> jnp.ndarray:
        current = self._sc_inverse_barrier_jax(self._sc_barrier_values_jax(state))
        observed_next = self._sc_inverse_barrier_jax(self._sc_barrier_values_jax(next_state))
        return observed_next - self.config.sc_barrier_gamma * (beta - current)

    def _sc_barrier_values_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.ones(1, dtype=float)
        values = self._projection_barrier_values_with_margin_jax(state)
        if self.config.sc_use_combined_barrier:
            return jnp.array([jnp.min(values)], dtype=float)
        return values

    def _sc_inverse_barrier_jax(self, h: jnp.ndarray) -> jnp.ndarray:
        return 1.0 / jnp.maximum(h, self.config.sc_barrier_eps)

    def computeSafeFeedback(self, state: jnp.ndarray, control_sequence: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Return U_safe, xbar_ref, and K_BaS for SC-MPPI's embedded barrier-state sampler."""
        return self._compute_safe_feedback_jax(
            jnp.asarray(state, dtype=float),
            jnp.asarray(control_sequence, dtype=float),
        )

    def compute_safe_feedback(self, state: jnp.ndarray, control_sequence: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        return self.computeSafeFeedback(state, control_sequence)

    def _sc_beta_dim_static(self) -> int:
        if self.num_barriers == 0 or self.config.sc_use_combined_barrier:
            return 1
        return self.num_barriers

    def _sc_xbar_dim_static(self) -> int:
        return self.robot.state_dim + self._sc_beta_dim_static()

    def _sc_initial_xbar_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._sc_xbar_jax(state, self._sc_initial_beta_jax(state))

    def _sc_xbar_jax(self, state: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate((state, beta))

    def _sc_split_xbar_jax(self, xbar: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return xbar[: self.robot.state_dim], xbar[self.robot.state_dim :]

    def _sc_augmented_step_jax(self, xbar: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        state, beta = self._sc_split_xbar_jax(xbar)
        u = self.robot.clip_control(control)
        next_state = self.robot.step(state, u, self.config.dt)
        next_beta = self._sc_next_beta_jax(beta, state, next_state)
        return self._sc_xbar_jax(next_state, next_beta)

    def _sc_rollout_augmented_jax(self, state: jnp.ndarray, controls: jnp.ndarray) -> jnp.ndarray:
        xbar0 = self._sc_initial_xbar_jax(state)

        def step_scan(xbar, control):
            xbar_next = self._sc_augmented_step_jax(xbar, control)
            return xbar_next, xbar_next

        _xbar_final, xbars_next = jax.lax.scan(step_scan, xbar0, controls)
        return jnp.concatenate((xbar0[None, :], xbars_next), axis=0)

    def _sc_linearize_augmented_step_jax(
        self,
        xbar: jnp.ndarray,
        control: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        eps = 1e-4
        xbar_perturbations = eps * jnp.eye(self._sc_xbar_dim_static(), dtype=float)
        control_perturbations = eps * jnp.eye(self.robot.control_dim, dtype=float)
        state_diff = lambda delta: (
            self._sc_augmented_step_jax(xbar + delta, control)
            - self._sc_augmented_step_jax(xbar - delta, control)
        ) / (2.0 * eps)
        control_diff = lambda delta: (
            self._sc_augmented_step_jax(xbar, control + delta)
            - self._sc_augmented_step_jax(xbar, control - delta)
        ) / (2.0 * eps)
        a_matrix = jax.vmap(state_diff)(xbar_perturbations).T
        b_matrix = jax.vmap(control_diff)(control_perturbations).T
        return a_matrix, b_matrix

    def _sc_feedback_cost_matrices_jax(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        beta_dim = self._sc_beta_dim_static()
        q_diag = jnp.concatenate(
            (
                self.config.sc_state_cost_weight * jnp.ones(self.robot.state_dim, dtype=float),
                self.config.sc_beta_cost_weight * jnp.ones(beta_dim, dtype=float),
            )
        )
        q_matrix = jnp.diag(q_diag)
        r_matrix = self.config.sc_control_cost_weight * jnp.eye(self.robot.control_dim, dtype=float)
        beta_target = jnp.ones(beta_dim, dtype=float) / jnp.maximum(
            self.config.br_clearance_margin,
            self.config.sc_barrier_eps,
        )
        return q_matrix, r_matrix, beta_target

    def _sc_backward_pass_jax(
        self,
        xbar_ref: jnp.ndarray,
        safe_sequence: jnp.ndarray,
        nominal_sequence: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        q_matrix, r_matrix, beta_target = self._sc_feedback_cost_matrices_jax()
        regularization = self.config.sc_feedback_regularization * jnp.eye(self.robot.control_dim, dtype=float)

        def state_cost_gradient(xbar: jnp.ndarray) -> jnp.ndarray:
            target = jnp.concatenate((xbar[: self.robot.state_dim], beta_target))
            return q_matrix @ (xbar - target)

        terminal_v_x = state_cost_gradient(xbar_ref[-1])
        terminal_v_xx = q_matrix

        def backward_scan(carry, inputs):
            v_x_next, v_xx_next = carry
            xbar_k, u_k, u_nominal_k = inputs
            a_matrix, b_matrix = self._sc_linearize_augmented_step_jax(xbar_k, u_k)
            l_x = state_cost_gradient(xbar_k)
            l_u = r_matrix @ (u_k - u_nominal_k)
            q_x = l_x + a_matrix.T @ v_x_next
            q_u = l_u + b_matrix.T @ v_x_next
            q_xx = q_matrix + a_matrix.T @ v_xx_next @ a_matrix
            q_ux = b_matrix.T @ v_xx_next @ a_matrix
            q_uu = r_matrix + b_matrix.T @ v_xx_next @ b_matrix + regularization
            q_u_solution = self._solve_with_pinv_fallback(q_uu, q_u, q_uu)
            q_ux_solution = self._solve_with_pinv_fallback(q_uu, q_ux, q_uu)
            feedforward = -q_u_solution
            feedback_gain = -q_ux_solution
            v_x = q_x - q_ux.T @ q_u_solution
            v_xx = q_xx - q_ux.T @ q_ux_solution
            v_xx = 0.5 * (v_xx + v_xx.T)
            return (v_x, v_xx), (feedforward, feedback_gain)

        (_v_x0, _v_xx0), (feedforward_reversed, gains_reversed) = jax.lax.scan(
            backward_scan,
            (terminal_v_x, terminal_v_xx),
            (
                xbar_ref[:-1][::-1],
                safe_sequence[::-1],
                nominal_sequence[::-1],
            ),
        )
        return gains_reversed[::-1], feedforward_reversed[::-1]

    def _compute_safe_feedback_jax(
        self,
        state: jnp.ndarray,
        control_sequence: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        nominal_sequence = self.robot.clip_controls(control_sequence)
        if self.num_barriers == 0:
            xbar_ref = self._sc_rollout_augmented_jax(state, nominal_sequence)
            gains = jnp.zeros(
                (self.config.horizon, self.robot.control_dim, self._sc_xbar_dim_static()),
                dtype=float,
            )
            return nominal_sequence, xbar_ref, gains

        def feedback_iteration(safe_sequence, _unused):
            xbar_ref = self._sc_rollout_augmented_jax(state, safe_sequence)
            _gains, feedforward = self._sc_backward_pass_jax(xbar_ref, safe_sequence, nominal_sequence)
            feedforward = jnp.clip(feedforward, -self.config.sc_feedback_clip, self.config.sc_feedback_clip)
            return self.robot.clip_controls(safe_sequence + feedforward), None

        if self.config.sc_feedback_iterations <= 0:
            safe_sequence = nominal_sequence
        else:
            safe_sequence, _unused = jax.lax.scan(
                feedback_iteration,
                nominal_sequence,
                jnp.arange(self.config.sc_feedback_iterations),
            )
        xbar_ref = self._sc_rollout_augmented_jax(state, safe_sequence)
        gains, _feedforward = self._sc_backward_pass_jax(xbar_ref, safe_sequence, nominal_sequence)
        return safe_sequence, xbar_ref, gains

    def _gs_constraint_values_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.ones(1, dtype=float)
        return self._projection_barrier_values_with_margin_jax(state) - self.config.gs_composite_margin

    def _gs_composite_barrier_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        values = self._gs_constraint_values_jax(state)
        rho = self.config.gs_softmin_rho
        scaled = -rho * values
        stable = scaled - jnp.max(scaled)
        return -(jnp.log(jnp.sum(jnp.exp(stable))) + jnp.max(scaled)) / rho

    def _gs_composite_barrier_jacobian_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        h_base = self._gs_composite_barrier_jax(state)
        eps = 1e-4
        perturbations = eps * jnp.eye(self.robot.state_dim, dtype=float)
        diff_fn = lambda delta: (self._gs_composite_barrier_jax(state + delta) - h_base) / eps
        return jax.vmap(diff_fn)(perturbations)

    def _gs_safe_control_jax(self, state: jnp.ndarray, desired_control: jnp.ndarray) -> jnp.ndarray:
        desired = self.robot.clip_control(desired_control)
        if self.num_barriers == 0:
            return desired
        h = self._gs_composite_barrier_jax(state)
        grad_h = self._gs_composite_barrier_jacobian_jax(state)
        lf_h = jnp.dot(grad_h, self.robot.drift(state))
        lg_h = grad_h @ self.robot.control_matrix(state)
        omega = lf_h + jnp.dot(lg_h, desired) + self.config.gs_alpha * h
        numerator = jnp.maximum(-omega, 0.0)
        denominator = jnp.dot(lg_h, lg_h) + h**2 / jnp.maximum(self.config.gs_closed_form_gamma, 1e-9) + 1e-9
        safe = desired + lg_h * numerator / denominator
        # The GS-MPPI theorem assumes the closed-form safe control is applied as computed.
        # This repo clips to actuator bounds for fair robot-model comparisons, so the
        # formal CBF guarantee is not retained when clipping changes the control.
        return self.robot.clip_control(safe)

    def _gs_cbf_residual_jax(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.array(0.0, dtype=float)
        u = self.robot.clip_control(control)
        h = self._gs_composite_barrier_jax(state)
        grad_h = self._gs_composite_barrier_jacobian_jax(state)
        lf_h = jnp.dot(grad_h, self.robot.drift(state))
        lg_h = grad_h @ self.robot.control_matrix(state)
        omega = lf_h + jnp.dot(lg_h, u) + self.config.gs_alpha * h
        return jnp.maximum(-omega, 0.0)

    def _clearance(self, state: jnp.ndarray) -> float:
        return float(jax.device_get(self._clearance_jax(jnp.asarray(state, dtype=float))))

    def _clearance_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        points = self.robot.body_points(state)
        distances = self.obstacle_field.signed_distance(points)
        return jnp.min(distances) - self.robot.body_point_radius

    def _barrier_values(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._barrier_values_jax(jnp.asarray(state, dtype=float))

    def _barrier_values_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        if self.sdf_model is not None:
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

        if self.sdf_model is not None:
            h, barrier_jacobian = self._neural_projection_barrier_values_and_jacobian_jax(state)
        else:
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

    def _neural_projection_barrier_values_and_jacobian_jax(
        self,
        state: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate neural barriers once and chain their exact local Jacobian."""
        barrier_state_fn = lambda source_state: self.robot.projection_barrier_state(
            source_state,
            self.config.dt,
        )
        barrier_state = barrier_state_fn(state)
        assert self.sdf_model is not None
        barrier_values, barrier_state_jacobian = self.sdf_model.obstacle_barriers_and_jacobian(
            barrier_state,
            self.obstacle_field,
        )
        projection_state_jacobian = jax.jacfwd(barrier_state_fn)(state)
        barrier_values = jnp.asarray(barrier_values, dtype=float)
        barrier_state_jacobian = jnp.asarray(barrier_state_jacobian, dtype=float)
        return (
            barrier_values - self.config.barrier_projection_margin,
            barrier_state_jacobian @ projection_state_jacobian,
        )

    def _projection_barrier_values(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_values_jax(jnp.asarray(state, dtype=float))

    def _projection_barrier_values_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        barrier_state = self.robot.projection_barrier_state(state, self.config.dt)
        return self._barrier_values_jax(barrier_state)

    def _projection_barrier_values_with_margin(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_values_with_margin_jax(jnp.asarray(state, dtype=float))

    def _projection_barrier_values_with_margin_jax(self, state: jnp.ndarray) -> jnp.ndarray:
        return self._projection_barrier_values_jax(state) - self.config.barrier_projection_margin

    def _uses_cbf_qp_static(self) -> bool:
        return self.algo == "mppi_cbf"

    def _uses_rollout_cbf_qp_static(self) -> bool:
        return False

    def _cbf_qp_filter_jax(self, state: jnp.ndarray, desired_control: jnp.ndarray) -> jnp.ndarray:
        if not self._uses_cbf_qp_static() or self.num_barriers == 0:
            return self.robot.clip_control(desired_control)
        a_matrix, b_vector = self._cbf_inequality_problem_jax(state)
        return self._solve_cbf_qp_jax(self.robot.clip_control(desired_control), a_matrix, b_vector)

    def _cbf_inequality_problem_jax(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = self._projection_barrier_values_with_margin_jax(state)
        barrier_jacobian = self._projection_barrier_jacobian_jax(state)
        drift = self.robot.drift(state)
        control_matrix = self.robot.control_matrix(state)
        a_matrix = barrier_jacobian @ control_matrix
        b_vector = -(barrier_jacobian @ drift + self.config.cbf_alpha * h)
        return a_matrix, b_vector

    def _solve_cbf_qp_jax(
        self,
        desired_control: jnp.ndarray,
        a_matrix: jnp.ndarray,
        b_vector: jnp.ndarray,
    ) -> jnp.ndarray:
        """ADMM solve for min ||u-u_des||^2 subject to CBF rows and box bounds."""
        control_dim = self.robot.control_dim
        lower = self.robot.control_bounds[:, 0]
        upper = self.robot.control_bounds[:, 1]
        eye = jnp.eye(control_dim, dtype=float)
        constraint_matrix = jnp.concatenate((a_matrix, eye, -eye), axis=0)
        constraint_bound = jnp.concatenate((b_vector, lower, -upper), axis=0)

        rho = self.config.cbf_qp_rho
        lhs = (
            eye
            + rho * (constraint_matrix.T @ constraint_matrix)
            + self.config.cbf_qp_regularization * eye
        )
        slack = jnp.maximum(constraint_matrix @ desired_control - constraint_bound, 0.0)
        dual = jnp.zeros_like(slack)

        def admm_step(carry, _unused):
            slack_k, dual_k = carry
            rhs = desired_control + rho * constraint_matrix.T @ (slack_k + constraint_bound - dual_k)
            control = jnp.linalg.solve(lhs, rhs)
            primal = constraint_matrix @ control - constraint_bound
            slack_next = jnp.maximum(primal + dual_k, 0.0)
            dual_next = dual_k + primal - slack_next
            return (slack_next, dual_next), control

        (_slack_final, _dual_final), controls = jax.lax.scan(
            admm_step,
            (slack, dual),
            jnp.arange(self.config.cbf_qp_iterations),
        )
        control = controls[-1]
        return jnp.clip(control, lower, upper)

    def _cbf_constraint_residual_jax(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        if self.num_barriers == 0:
            return jnp.array(0.0, dtype=float)
        a_matrix, b_vector = self._cbf_inequality_problem_jax(state)
        violation = b_vector - a_matrix @ self.robot.clip_control(control)
        return jnp.maximum(jnp.max(violation), 0.0)

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
