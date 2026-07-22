from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
for path in (REPO_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import basic_demo
from controller import MPPIConfig, MPPIController
import random_benchmark
from robots import create_robot
from sdf import PretrainedSDFUnavailable, default_obstacle_field, load_pretrained_sdf_for_robot


SUPPORTED_NSDF_ROBOTS = ("unicycle", "dynamic_unicycle", "planar_quadrotor")


@pytest.mark.parametrize("robot_name", SUPPORTED_NSDF_ROBOTS)
def test_pretrained_nsdf_has_finite_barrier_per_obstacle(robot_name: str) -> None:
    robot = create_robot(robot_name)
    field = default_obstacle_field(robot_name)
    model = load_pretrained_sdf_for_robot(robot_name)

    barriers = model.obstacle_barriers(robot.default_state, field)

    assert barriers.shape == (len(field.obstacles),)
    assert bool(jnp.all(jnp.isfinite(barriers)))


@pytest.mark.parametrize("robot_name", SUPPORTED_NSDF_ROBOTS)
def test_pretrained_nsdf_uses_float32_and_exact_state_jacobian(robot_name: str) -> None:
    robot = create_robot(robot_name)
    field = default_obstacle_field(robot_name)
    model = load_pretrained_sdf_for_robot(robot_name)
    state = robot.default_state.at[0].add(0.137).at[1].add(-0.083).at[2].add(0.071)

    values, direct_jacobian = model.obstacle_barriers_and_jacobian(state, field)
    autodiff_jacobian = jax.jacfwd(lambda candidate: model.obstacle_barriers(candidate, field))(state)

    parameter_dtypes = {
        parameter.dtype
        for layer in model.params["params"].values()
        for parameter in layer.values()
    }
    assert parameter_dtypes == {jnp.dtype(jnp.float32)}
    assert values.dtype == jnp.float32
    assert direct_jacobian.dtype == jnp.float32
    # GPU float32 dense products may use reduced-precision accumulation, so the
    # explicit backpropagation and XLA autodiff paths need a realistic tolerance.
    assert jnp.allclose(direct_jacobian, autodiff_jacobian, rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize("robot_name", SUPPORTED_NSDF_ROBOTS)
def test_controller_chains_nsdf_jacobian_through_projection_state(robot_name: str) -> None:
    robot = create_robot(robot_name)
    field = default_obstacle_field(robot_name)
    model = load_pretrained_sdf_for_robot(robot_name)
    controller = MPPIController(
        robot,
        field,
        algo="brmppi",
        sdf_model=model,
        config=MPPIConfig(horizon=2, samples=3, dt=0.1, plot_samples=0),
    )
    state = robot.default_state.at[0].add(0.137).at[1].add(-0.083).at[2].add(0.071)
    if robot_name == "dynamic_unicycle":
        state = state.at[3].set(0.6)
    elif robot_name == "planar_quadrotor":
        state = state.at[3].set(0.6).at[4].set(-0.4)

    _values, direct_jacobian = controller._neural_projection_barrier_values_and_jacobian_jax(state)
    autodiff_jacobian = jax.jacfwd(controller._projection_barrier_values_with_margin_jax)(state)

    assert direct_jacobian.shape == (len(field.obstacles), robot.state_dim)
    assert jnp.allclose(direct_jacobian, autodiff_jacobian, rtol=2e-2, atol=5e-3)


@pytest.mark.parametrize(
    ("robot_name", "inside_point", "outside_point"),
    (
        ("unicycle", (0.0, 0.0), (0.8, 0.0)),
        ("dynamic_unicycle", (0.0, 0.0), (0.8, 0.0)),
        ("planar_quadrotor", (0.0, 0.0), (0.6, 0.0)),
    ),
)
def test_pretrained_nsdf_has_expected_shape_sign(
    robot_name: str,
    inside_point: tuple[float, float],
    outside_point: tuple[float, float],
) -> None:
    model = load_pretrained_sdf_for_robot(robot_name)

    distances = model.signed_distance(jnp.asarray((inside_point, outside_point), dtype=float))

    assert float(distances[0]) < 0.0
    assert float(distances[1]) > 0.1


@pytest.mark.parametrize("robot_name", ("single_integrator", "mobile_arm"))
def test_pretrained_nsdf_reports_missing_geometry_checkpoint(robot_name: str) -> None:
    with pytest.raises(PretrainedSDFUnavailable):
        load_pretrained_sdf_for_robot(robot_name)


def test_brmppi_command_uses_neural_sdf() -> None:
    robot = create_robot("unicycle")
    field = default_obstacle_field(robot.name)
    model = load_pretrained_sdf_for_robot(robot.name)
    config = MPPIConfig(horizon=2, samples=3, dt=0.1, plot_samples=0)
    controller = MPPIController(robot, field, algo="brmppi", sdf_model=model, config=config, seed=7)

    action, diagnostics = controller.command(robot.default_state, robot.default_goal)

    assert action.shape == (robot.control_dim,)
    assert bool(jnp.all(jnp.isfinite(action)))
    assert controller.barrier_source == "neural_sdf"
    assert diagnostics["barrier_source"] == "neural_sdf"
    assert diagnostics["cbf_constraint_max_violation"] is None


def test_controller_rejects_nsdf_for_other_methods() -> None:
    robot = create_robot("unicycle")
    field = default_obstacle_field(robot.name)
    model = load_pretrained_sdf_for_robot(robot.name)

    with pytest.raises(NotImplementedError, match="only implemented for brmppi"):
        MPPIController(robot, field, algo="penalty_mppi", sdf_model=model)


def test_controller_rejects_legacy_nsdf_without_direct_jacobian() -> None:
    class LegacyBarrierModel:
        def obstacle_barriers(self, state, field):
            return jnp.zeros((len(field.obstacles),), dtype=state.dtype)

    robot = create_robot("unicycle")
    field = default_obstacle_field(robot.name)

    with pytest.raises(TypeError, match="obstacle_barriers_and_jacobian"):
        MPPIController(robot, field, algo="brmppi", sdf_model=LegacyBarrierModel())


def test_demo_defaults_to_analytic_sdf() -> None:
    assert basic_demo.load_selected_sdf("unicycle", "brmppi", nsdf=False) is None


def test_demo_rejects_nsdf_for_other_methods() -> None:
    with pytest.raises(NotImplementedError, match="only implemented for brmppi"):
        basic_demo.load_selected_sdf("unicycle", "penalty_mppi", nsdf=True)


def test_demo_reports_unsupported_nsdf_robot() -> None:
    with pytest.raises(NotImplementedError, match="single_integrator"):
        basic_demo.load_selected_sdf("single_integrator", "brmppi", nsdf=True)


def test_benchmark_uses_repository_tuned_brmppi_config_by_default() -> None:
    args = argparse.Namespace(robot="unicycle", horizon=None, samples=None, dt=None)

    configs = random_benchmark.resolve_benchmark_configs(args, ("brmppi",), use_tuned_config=True)

    assert configs["brmppi"] is not None
    assert configs["brmppi"].horizon == 36
    assert configs["brmppi"].samples == 240
    assert configs["brmppi"].plot_samples == 0


def test_benchmark_rejects_nsdf_for_other_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        random_benchmark,
        "parse_args",
        lambda: argparse.Namespace(nsdf=True, algo="penalty_mppi"),
    )

    with pytest.raises(NotImplementedError, match="only implemented for brmppi"):
        random_benchmark.main()


def test_benchmark_reports_unsupported_nsdf_robot() -> None:
    with pytest.raises(NotImplementedError, match="mobile_arm"):
        random_benchmark.load_benchmark_sdf("mobile_arm", nsdf=True)
