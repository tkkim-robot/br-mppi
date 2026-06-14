from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controller import ALGORITHMS, MPPIConfig, MPPIController, load_tuned_config
from robots import ROBOT_REGISTRY, create_robot
from sdf import default_obstacle_field


SAFETY_BASELINES = ("mppi_cbf", "shield_mppi", "sc_mppi", "gs_mppi")


def small_tuned_config(algo: str, **overrides):
    return load_tuned_config(
        "single_integrator",
        algo,
        overrides={
            "horizon": 2,
            "samples": 3,
            "dt": 0.1,
            "plot_samples": 0,
            **overrides,
        },
    )


def small_test_config(robot_name: str, algo: str, **overrides):
    if robot_name == "single_integrator":
        return small_tuned_config(algo, **overrides)
    return MPPIConfig(
        horizon=2,
        samples=3,
        dt=0.1,
        plot_samples=0,
        **overrides,
    )


def test_tuned_single_integrator_configs_cover_all_algorithms() -> None:
    for algo in ALGORITHMS:
        config = load_tuned_config("single_integrator", algo)
        assert config.horizon > 0
        assert config.samples > 0
        assert config.plot_samples == 0


@pytest.mark.parametrize("algo", SAFETY_BASELINES)
@pytest.mark.parametrize("robot_name", tuple(sorted(ROBOT_REGISTRY)))
def test_cbf_baseline_runs_one_jax_command_for_each_robot(algo: str, robot_name: str) -> None:
    robot = create_robot(robot_name)
    field = default_obstacle_field(robot.name)
    config = small_test_config(robot_name, algo, cbf_qp_iterations=12)
    controller = MPPIController(robot, field, algo=algo, config=config, seed=7)

    action, diagnostics = controller.command(robot.default_state, robot.default_goal)

    assert isinstance(action, jax.Array)
    assert action.shape == (robot.control_dim,)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert diagnostics["uses_cbf_qp"] is (algo == "mppi_cbf")
    assert diagnostics["uses_rollout_cbf_qp"] is False
    assert jnp.isfinite(jnp.asarray(diagnostics["cbf_qp_max_violation"]))
    assert jnp.isfinite(jnp.asarray(diagnostics["best_cost"]))
    assert diagnostics["sampled_trajectories"].shape == (0, config.horizon + 1, robot.state_dim)


def test_shield_mppi_uses_dcbf_rollout_cost_and_no_qp(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = create_robot("single_integrator")
    field = default_obstacle_field(robot.name)
    config = small_tuned_config("shield_mppi", horizon=1, samples=2, dt=0.4, shield_cbf_penalty_weight=100.0)
    shield = MPPIController(robot, field, algo="shield_mppi", config=config, seed=4)
    plain = MPPIController(robot, field, algo="mppi", config=config, seed=4)

    def forbidden_qp(*_args, **_kwargs):
        raise AssertionError("shield_mppi must not call the CBF-QP filter")

    monkeypatch.setattr(shield, "_cbf_qp_filter_jax", forbidden_qp)
    state = jnp.array([-7.15, -1.55], dtype=float)
    goal = jnp.array([-4.0, -1.55], dtype=float)
    controls = jnp.array([[1.25, 0.0]], dtype=float)

    shield_cost = shield._rollout(state, goal, controls).cost
    plain_cost = plain._rollout(state, goal, controls).cost
    assert shield_cost > plain_cost

    action, diagnostics = shield.command(state, goal)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert diagnostics["uses_cbf_qp"] is False
    assert diagnostics["uses_rollout_cbf_qp"] is False
    assert diagnostics["uses_shield_repair"] is True
    assert diagnostics["shield_repair_violation_after"] <= diagnostics["shield_repair_violation_before"] + 1e-9


def test_shield_mppi_repair_does_not_increase_dcbf_violation() -> None:
    robot = create_robot("single_integrator")
    field = default_obstacle_field(robot.name)
    config = small_tuned_config(
        "shield_mppi",
        horizon=4,
        samples=2,
        dt=0.25,
        shield_alpha=0.98,
        shield_repair_horizon=4,
        shield_repair_steps=12,
        shield_repair_step_size=0.08,
    )
    controller = MPPIController(robot, field, algo="shield_mppi", config=config, seed=3)
    state = jnp.array([-7.15, -1.55], dtype=float)
    sequence = jnp.tile(jnp.array([1.25, 0.0], dtype=float)[None, :], (config.horizon, 1))

    before = controller._shield_repair_violation_jax(state, sequence)
    repaired = controller._shield_repair_sequence_jax(state, sequence)
    after = controller._shield_repair_violation_jax(state, repaired)

    assert before > 0.0
    assert after <= before + 1e-9


def test_sc_mppi_compute_safe_feedback_shapes_and_nonzero_gain() -> None:
    robot = create_robot("single_integrator")
    field = default_obstacle_field(robot.name)
    config = small_tuned_config("sc_mppi", horizon=3, samples=4, dt=0.2, sc_feedback_iterations=1)
    sc = MPPIController(robot, field, algo="sc_mppi", config=config, seed=9)
    state = jnp.array([-7.15, -1.55], dtype=float)
    goal = jnp.array([7.0, 3.6], dtype=float)
    base = jnp.tile(robot.nominal_control(state, goal)[None, :], (config.horizon, 1))

    u_safe, xbar_ref, gains = sc.computeSafeFeedback(state, base)

    assert u_safe.shape == (config.horizon, robot.control_dim)
    assert xbar_ref.shape == (config.horizon + 1, robot.state_dim + 1)
    assert gains.shape == (config.horizon, robot.control_dim, robot.state_dim + 1)
    assert bool(jnp.all(jnp.isfinite(u_safe)))
    assert bool(jnp.all(jnp.isfinite(xbar_ref)))
    assert bool(jnp.all(jnp.isfinite(gains)))
    assert jnp.linalg.norm(gains) > 0.0


def test_sc_mppi_sampling_uses_dbas_gain_not_qp(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = create_robot("single_integrator")
    field = default_obstacle_field(robot.name)
    config = small_tuned_config("sc_mppi", samples=4, dt=0.2, sc_feedback_iterations=1)
    sc = MPPIController(robot, field, algo="sc_mppi", config=config, seed=9)
    plain = MPPIController(robot, field, algo="mppi", config=config, seed=9)
    state = jnp.array([-7.15, -1.55], dtype=float)
    goal = jnp.array([7.0, 3.6], dtype=float)
    base = jnp.tile(robot.nominal_control(state, goal)[None, :], (config.horizon, 1))
    u_safe, xbar_ref, gains = sc.computeSafeFeedback(state, base)
    candidates, _ = sc._sample_control_sequences_jax(u_safe, sc.key)

    def forbidden_qp(*_args, **_kwargs):
        raise AssertionError("sc_mppi must not call the CBF-QP filter")

    monkeypatch.setattr(sc, "_cbf_qp_filter_jax", forbidden_qp)

    _sc_cost, _sc_states, sc_controls, *_ = sc._rollout_jax(
        state,
        goal,
        candidates[1],
        sc.alpha_state,
        xbar_ref,
        gains,
    )
    plain_controls = plain._rollout(state, goal, candidates[1]).controls
    assert jnp.linalg.norm(sc_controls - plain_controls) > 0.0

    action, diagnostics = sc.command(state, goal)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert diagnostics["uses_cbf_qp"] is False
    assert diagnostics["sc_feedback_model"] == "dbas_ilqr_feedback"
    assert diagnostics["sc_feedback_norm"] > 0.0


def test_gs_mppi_uses_composite_barrier_and_closed_form_control(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = create_robot("single_integrator")
    field = default_obstacle_field(robot.name)
    config = small_tuned_config("gs_mppi", dt=0.2)
    controller = MPPIController(robot, field, algo="gs_mppi", config=config, seed=5)

    def forbidden_qp(*_args, **_kwargs):
        raise AssertionError("gs_mppi must not use the generic ADMM CBF-QP")

    monkeypatch.setattr(controller, "_solve_cbf_qp_jax", forbidden_qp)
    state = jnp.array([-7.15, -1.55], dtype=float)
    desired = jnp.array([1.25, 0.0], dtype=float)

    composite = controller._gs_composite_barrier_jax(state)
    safe = controller._gs_safe_control_jax(state, desired)

    assert jnp.shape(composite) == ()
    assert jnp.linalg.norm(safe - robot.clip_control(desired)) > 0.0
    assert controller._uses_cbf_qp_static() is False

    action, diagnostics = controller.command(state, robot.default_goal)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert diagnostics["uses_cbf_qp"] is False
    assert diagnostics["uses_gs_closed_form_safe_control"] is True
    assert jnp.isfinite(jnp.asarray(diagnostics["gs_cbf_residual_after_clipping"]))
