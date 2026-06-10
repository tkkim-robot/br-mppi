# Hyperparameter Tuning Handover

This repo provides two overnight-oriented scripts:

- `examples/random_benchmark.py`: evaluates one or more methods on paired randomized obstacle fields.
- `examples/tune_hyperparameters.py`: runs Optuna hyperparameter tuning and logs metrics/configs to W&B.

The intended workflow is:

1. Tune `brmppi` first for one dynamics until it reaches the highest possible success rate, ideally 100%.
2. Save the best BR-MPPI config.
3. For each baseline on the same dynamics, reuse the BR-tuned shared MPPI parameters and tune only method-specific leftover parameters.
4. Repeat for every dynamics.
5. Store final best-parameter benchmark results as markdown tables.

## Environment

Clone the repo on the overnight machine and install with uv:

```bash
git clone <repo-url>
cd br-mppi
uv sync
```

`pyproject.toml` includes `optuna` and `wandb`. `uv.lock` is intentionally ignored, so the overnight machine can resolve a CUDA-compatible environment locally.

## JAX CUDA

For GPU runs, first install a CUDA-enabled JAX build on the target machine. Follow the current JAX CUDA wheel instructions for the installed CUDA version. For many CUDA 12 systems this looks like:

```bash
uv pip install -U "jax[cuda12]"
```

Then verify JAX sees the GPU:

```bash
uv run python - <<'PY'
import jax
print(jax.devices())
PY
```

If this still prints only CPU devices, fix CUDA/JAX before running the overnight study. The controller is written in JAX and uses `lax.scan`/`vmap`, so GPU acceleration should help most for large sample counts.

## W&B

Log in once:

```bash
uv run wandb login
```

For offline logging, add:

```bash
--wandb-mode offline
```

To disable W&B entirely for a dry run:

```bash
--no-wandb
```

## Random Benchmark

The randomized benchmark uses fixed start/goal states from each robot model and randomized circular obstacles.

By default, the obstacle count matches the standard demo scene:

- `single_integrator`: 20
- `unicycle`: 20
- `dynamic_unicycle`: 20
- `planar_quadrotor`: 20
- `mobile_arm`: 16

Every algorithm in a run sees the same obstacle field per trial:

```text
field_seed = --seed + trial_index
controller_seed = --controller-seed + trial_index
```

Run a benchmark:

```bash
uv run python examples/random_benchmark.py \
  --robot unicycle \
  --algo all \
  --trials 100
```

The benchmark stops a trial immediately on collision, stops successfully when the goal is reached, and otherwise times out at a generous dynamics-specific cap.

## Tuning Objective

The Optuna objective is success-rate first:

```text
success = reached goal without collision
success_rate = successes / 100 randomized benchmark trials
```

For equal success rates, the script adds a very small travel-time tie-breaker based on average successful travel time. The tie-breaker is deliberately smaller than one success out of 100 trials, so it cannot dominate success rate.

Trials are pruned with Optuna `MedianPruner` using partial benchmark success rates every `--prune-interval` trials.

## Step 1: Tune BR-MPPI

Tune BR-MPPI first for a single dynamics. This search includes shared MPPI parameters and BR-specific parameters:

- MPPI horizon
- MPPI parallel samples, searched from 100 to 1000
- temperature
- control noise covariance scale
- goal/final/control weights
- projection margin
- alpha initial/rate/state bounds
- alpha noise scale
- alpha projection inverse weight
- barrier buffer and BR clearance/collision costs

Example for unicycle:

```bash
uv run python examples/tune_hyperparameters.py \
  --robot unicycle \
  --algo brmppi \
  --optuna-trials 100 \
  --benchmark-trials 100 \
  --final-eval-trials 100 \
  --wandb-project br-mppi-tuning \
  --study-name unicycle_brmppi
```

The output directory contains:

- `output/tuning/unicycle_brmppi.db`
- `output/tuning/unicycle_brmppi_best_config.json`
- `output/tuning/unicycle_brmppi_study.json`
- `output/tuning/unicycle_brmppi_best_benchmark.json`
- `output/tuning/unicycle_brmppi_best_benchmark.md`

If BR-MPPI cannot reach 100% success no matter what, use the highest-success-rate config and proceed.

If you pass `--horizon` or `--samples`, those values are treated as fixed overrides and are not tuned. Omit those flags for the normal overnight BR-MPPI search, where `samples` is searched from 100 to 1000.

## Step 2: Tune Baselines With Shared Params Fixed

After BR-MPPI is tuned for a dynamics, use its best config as the shared baseline config for the other methods.

Example:

```bash
uv run python examples/tune_hyperparameters.py \
  --robot unicycle \
  --algo shield_mppi \
  --fixed-config output/tuning/unicycle_brmppi_best_config.json \
  --no-tune-shared \
  --optuna-trials 60 \
  --benchmark-trials 100 \
  --final-eval-trials 100 \
  --wandb-project br-mppi-tuning \
  --study-name unicycle_shield_mppi
```

Repeat for:

```text
mppi
penalty_mppi
mppi_cbf
shield_mppi
sc_mppi
gs_mppi
```

For methods with no extra parameters, such as plain `mppi`, the script still evaluates the fixed shared config and writes final benchmark files.

## Step 3: Repeat For All Dynamics

Repeat the BR-first sequence for:

```text
single_integrator
unicycle
dynamic_unicycle
planar_quadrotor
mobile_arm
```

Suggested overnight layout:

```bash
for robot in single_integrator unicycle dynamic_unicycle planar_quadrotor mobile_arm; do
  uv run python examples/tune_hyperparameters.py \
    --robot "$robot" \
    --algo brmppi \
    --optuna-trials 100 \
    --benchmark-trials 100 \
    --final-eval-trials 100 \
    --study-name "${robot}_brmppi"
done
```

Then tune baselines for each dynamics using each dynamics' BR best config.

## Final Tables

Each tuning run writes one markdown table:

```text
output/tuning/<study-name>_best_benchmark.md
```

Use these markdown tables to collect final results per dynamics. The table columns include:

- reached count
- collision count
- timeout count
- success rate
- collision rate
- mean steps
- mean command time
- worst clearance

The corresponding JSON file includes the full per-trial records and the exact best config.

## Quick Smoke Test

Before launching overnight, run a tiny smoke test:

```bash
uv run python examples/tune_hyperparameters.py \
  --robot single_integrator \
  --algo brmppi \
  --optuna-trials 2 \
  --benchmark-trials 2 \
  --final-eval-trials 2 \
  --max-steps 12 \
  --horizon 3 \
  --samples 4 \
  --no-wandb \
  --study-name smoke_single_integrator_brmppi
```

This should finish quickly and create study/config/benchmark outputs under `output/tuning/`.
