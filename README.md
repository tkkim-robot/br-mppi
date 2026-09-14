# Barrier-Rate MPPI

This repository implements Barrier-Rate guided Model Predictive Path Integral control (BR-MPPI) for obstacle avoidance. BR-MPPI samples control inputs together with barrier-rate variables and projects the sampled rollouts through the barrier constraints before selecting the control to execute.

## Features

- BR-MPPI plus six MPPI-family baselines: MPPI, Penalty MPPI, MPPI-CBF, Shield-MPPI, SC-MPPI, and GS-MPPI.
- Five robot models: single integrator, unicycle, dynamic unicycle, planar quadrotor, and a planar mobile arm.
- Analytic signed-distance barriers and bundled pretrained neural-SDF checkpoints.
- Interactive demos, fixed-scene method comparisons, randomized benchmarks, tuning tools, and regression tests.

## Installation

Install [uv](https://docs.astral.sh/uv/) and create the environment:

```bash
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
- `tests/`: regression tests, deterministic NumPy reference traces, and the five-method comparison.

Generated plots, videos, logs, benchmark reports, and tuning studies are written under `output/` and excluded from version control. The JSON files in `tests/baselines/` are regression fixtures; those in `sdf/trained_models/` contain checkpoint metadata required by the neural-SDF loader.

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

Run the regression suite:

```bash
uv run pytest
```

## Hyperparameter tuning

`examples/tune_hyperparameters.py` requires a CUDA-enabled JAX installation and an NVIDIA GPU. Tune BR-MPPI first, then reuse its shared MPPI parameters when tuning the other methods:

```bash
uv run python examples/tune_hyperparameters.py \
  --robot unicycle --algo brmppi \
  --optuna-trials 100 --benchmark-trials 100 --final-eval-trials 100 \
  --study-name unicycle_brmppi --no-wandb

uv run python examples/tune_hyperparameters.py \
  --robot unicycle --algo shield_mppi \
  --fixed-config output/tuning/unicycle_brmppi_best_config.json \
  --no-tune-shared --study-name unicycle_shield_mppi --no-wandb
```

Each study writes its database, best configuration, trial records, and final benchmark reports to `output/tuning/`. The checked-in configurations used by demos and benchmarks are in `configs/tuned_hyperparameters.yaml`. To enable W&B logging, run `uv run wandb login` and omit `--no-wandb`; `--wandb-mode offline` stores logs locally.

After generating `<robot>_<method>_best_benchmark.json` reports for all five robots and six safety methods, plot their outcomes and command times:

```bash
uv run python examples/plot_benchmark_summary.py
```

Add `--include-mppi` to include plain MPPI reports. Use each script's `--help` for output paths and other options.

## Neural-SDF training

Train a rectangle checkpoint:

```bash
uv run python examples/train_nsdf.py --preset link1 --seed 0
```

Available presets are `link1`, `link7`, `mobile_base`, and `mobile_link`. Training writes a new checkpoint and validation metadata under `output/nsdf_training/`. Bundled checkpoints remain available under `sdf/trained_models/`; replacing them also requires updating the loader's pinned checksums.

The mobile-arm neural barrier combines learned base/link distances with an analytic rectangle guard. The single integrator uses only analytic barriers.
