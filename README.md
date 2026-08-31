# Barrier-Rate MPPI

This repository implements Barrier-Rate guided Model Predictive Path Integral control (BR-MPPI) for obstacle avoidance. BR-MPPI samples control inputs together with barrier-rate variables and projects the sampled rollouts through the barrier constraints before selecting the control to execute.

## Features

- BR-MPPI plus six MPPI-family baselines: MPPI-CBF, Shield-MPPI, SC-MPPI, and GS-MPPI.
- Five robot models: single integrator, unicycle, dynamic unicycle, planar quadrotor, and a planar mobile arm.
- Analytic signed-distance barriers and bundled pretrained neural-SDF checkpoints.
- Interactive demos, fixed-scene method comparisons, randomized benchmarks, tuning tools, and regression tests.

## Installation

Install [uv](https://docs.astral.sh/uv/), clone the repository, and create the environment:

```bash
git clone https://github.com/tkkim-robot/br-mppi.git
cd br-mppi
uv sync --extra dev
```

Python 3.10 or newer is required. Saving MP4 animations also requires `ffmpeg` on `PATH`.

## Repository layout

- `controller/`: BR-MPPI and the MPPI-family baseline controllers.
- `robots/`: robot dynamics, geometry, limits, and visualization.
- `sdf/`: analytic obstacle geometry, neural-SDF inference, and pretrained checkpoints.
- `configs/`: tuned controller parameters for each method and robot.
- `examples/`: single-run demos, benchmarks, NSDF training, and hyperparameter tuning.
- `tests/`: regression tests, stored NumPy baselines, and the five-method comparison.
- `docs/`: benchmark results and the detailed tuning workflow.

## Single demo

Run the default BR-MPPI unicycle example:

```bash
uv run python examples/basic_demo.py
```

The run opens an interactive animation and writes a summary figure to `output/uni_brmppi.png`. The animation shows the obstacle field, executed trajectory, sampled rollouts, and current best rollout.

Choose a controller, robot, and rollout settings from the command line:

```bash
uv run python examples/basic_demo.py \
  --algo mppi_cbf \
  --robot dynamic_unicycle \
  --samples 256 \
  --horizon 30 \
  --steps 300 \
  --seed 3
```

Controller choices are `brmppi`, `mppi`, `penalty_mppi`, `mppi_cbf`, `shield_mppi`, `sc_mppi`, and `gs_mppi`. Robot choices are `single_integrator`, `unicycle`, `dynamic_unicycle`, `planar_quadrotor`, and `mobile_arm`.

Run without a window, choose the summary path, or save an MP4:

```bash
uv run python examples/basic_demo.py --headless --save output/my_run.png
uv run python examples/basic_demo.py --save-animation --headless
```

Use a neural SDF as the BR-MPPI barrier:

```bash
uv run python examples/basic_demo.py --algo brmppi --robot unicycle --nsdf
```

Neural barriers are available for `unicycle`, `dynamic_unicycle`, `planar_quadrotor`, and `mobile_arm`; `single_integrator` currently uses the analytic barrier only. Run `uv run python examples/basic_demo.py --help` for all output and animation options.

## Five-method comparison

Run BR-MPPI and four tuned safety baselines in the same fixed obstacle field:

```bash
uv run python tests/test_compare_vis.py --dynamics unicycle
```

This opens a synchronized comparison of BR-MPPI, MPPI-CBF, Shield-MPPI, SC-MPPI, and GS-MPPI. The supported dynamics are the same five robot choices listed above. Controller settings come from `configs/tuned_hyperparameters.yaml`; use `--max-steps`, `--plot-samples`, or `--seed` to adjust the run.

Save the final frame and synchronized video without opening the GUI:

```bash
uv run python tests/test_compare_vis.py --dynamics unicycle \
  --save-figure --save-video --headless
```

Default comparison outputs are written to `output/compare_vis/`.

## Benchmarks and tests

Run the compact analytic/neural safety check:

```bash
uv run python examples/sanity_check.py --robot unicycle --algo brmppi --include-nsdf
```

Evaluate the tuned BR-MPPI configuration on randomized obstacle fields:

```bash
uv run python examples/random_benchmark.py \
  --robot unicycle --algo brmppi --trials 100
```
