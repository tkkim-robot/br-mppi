# Barrier-Rate MPPI

This repository is a dedicated home for Barrier-Rate guided Model Predictive Path Integral control (BR-MPPI) experiments.

The code is organized in the same style as `dpcbf`: local controllers, robot models, SDF utilities, and runnable examples. The old `mobile_arm/` checkout is intentionally not part of this repo and is gitignored; the pretrained checkpoints needed by the demos are vendored directly under `sdf/trained_models/`.

## Repository Layout

- `controller/`: MPPI, penalty MPPI, and barrier-rate guided MPPI
- `robots/`: single-integrator, unicycle, dynamic-unicycle, planar-quadrotor, and mobile-arm models
- `sdf/`: analytic obstacle signed-distance functions, pretrained neural SDF loaders, and vendored checkpoints
- `examples/basic_demo.py`: runnable demo with algorithm, robot, and NSDF flags
- `output/`: generated demo plots

## Setup

Install with `uv`; no Docker setup is required.

```bash
uv sync
```

## Run A Demo

Run the default BR-MPPI demo. It simulates the rollout, writes a summary PNG under
`output/`, and opens an interactive Matplotlib animation window without saving an MP4:

```bash
uv run python examples/basic_demo.py
```

Run headlessly and save only the summary plot:

```bash
uv run python examples/basic_demo.py --headless
```

Save an MP4 rollout animation:

```bash
uv run python examples/basic_demo.py --save-animation --headless
```

Animations are written to `output/animations/` by default using robot shorthands:
`si`, `uni`, `du`, `quad2d`, and `mobile_arm`. For example,
`output/animations/uni_brmppi.mp4` and `output/animations/uni_brmppi_nsdf.mp4`.
You can override the path or reduce frames:

```bash
uv run python examples/basic_demo.py --save-animation --animation-path output/animations/my_run.mp4 --animation-stride 3 --headless
```

### Regenerate Analytical Review Videos

The current BR-MPPI review videos use the analytic h-function/SDF path. Do not pass
`--nsdf` for these runs. The commands below regenerate the per-dynamics review
animations in `output/animations/`:

```bash
uv run python examples/basic_demo.py --algo brmppi --robot single_integrator --save-animation --headless
uv run python examples/basic_demo.py --algo brmppi --robot unicycle --save-animation --headless
uv run python examples/basic_demo.py --algo brmppi --robot dynamic_unicycle --save-animation --headless
uv run python examples/basic_demo.py --algo brmppi --robot planar_quadrotor --save-animation --headless
uv run python examples/basic_demo.py --algo brmppi --robot mobile_arm --save-animation --headless
```

The default simulation cap is generous, 500 steps, now that the JAX controller
can evaluate the sampled rollouts quickly. Successful runs terminate early as
soon as the robot reaches its goal tolerance, so the larger cap prevents clipped
review videos without making already-successful runs continue unnecessarily.

Choose the controller:

```bash
uv run python examples/basic_demo.py --algo brmppi
uv run python examples/basic_demo.py --algo mppi
uv run python examples/basic_demo.py --algo penalty_mppi
uv run python examples/basic_demo.py --algo mppi_cbf
uv run python examples/basic_demo.py --algo shield_mppi
uv run python examples/basic_demo.py --algo sc_mppi
uv run python examples/basic_demo.py --algo gs_mppi
```

The MPPI-family safety baselines are:

- `mppi`: vanilla MPPI with goal and control-effort costs.
- `penalty_mppi`: vanilla MPPI plus soft clearance/collision penalties.
- `mppi_cbf`: vanilla MPPI followed by a CBF-QP safety filter on the executed action.
- `shield_mppi`: Shield-MPPI style baseline. Rollouts use ordinary sampled MPPI controls, add a discrete-time CBF violation cost `C max(alpha h(x_k) - h(x_{k+1}), 0)`, then run a fixed-step JAX gradient repair on the first controls of the MPPI-updated sequence before executing the repaired first control.
- `sc_mppi`: SC-MPPI baseline with a finite-horizon `computeSafeFeedback(x0, U)` path. It embeds inverse barrier state `beta = B(h(x))`, rolls out an augmented `[x, beta]` reference trajectory, computes DBaS/iLQR-style feedback gains from trajectory linearization and a Riccati backward pass, and samples with `u_sample = U_safe + noise + K_BaS (xbar - xbar_ref)`.
- `gs_mppi`: GS-MPPI / MPPI-CBF style baseline. It builds a scalar soft-min composite CBF, rolls out the closed-form minimum-intervention safe control, updates the desired-control sequence from desired-control samples, and executes the closed-form safe version of the best sampled desired control. The implementation uses one planning time step rather than the paper's optional finer inner safety loop. It also clips the closed-form control to actuator bounds; that clipping can break the CBF inequality, so the paper's formal safety guarantee is not retained under clipping.
- `brmppi`: barrier-rate MPPI with sampled alpha rates and the closed-form BR projection.

Choose the robot:

```bash
uv run python examples/basic_demo.py --robot single_integrator
uv run python examples/basic_demo.py --robot unicycle
uv run python examples/basic_demo.py --robot dynamic_unicycle
uv run python examples/basic_demo.py --robot planar_quadrotor
uv run python examples/basic_demo.py --robot mobile_arm
```

Use a vendored pretrained neural signed-distance model for the barrier function:

```bash
uv run python examples/basic_demo.py --algo brmppi --robot unicycle --nsdf --headless
```

`--nsdf` is implemented only for `brmppi`; selecting it with another method raises
`NotImplementedError`. Analytic and neural demo runs use the same horizon, sample
count, environment, and seed so the flag changes only the h-function source.

Pretrained NSDF support is enabled only when a vendored `sdf/trained_models`
checkpoint matches the robot footprint used here:

- `unicycle`: `link1_model_4_16.npy`, rectangle `1.0 x 0.4`
- `dynamic_unicycle`: `link1_model_4_16.npy`, rectangle `1.0 x 0.4`
- `planar_quadrotor`: `link7_model_4_16.npy`, quad2 `0.56 x 0.28`

`single_integrator` and `mobile_arm` raise `NotImplementedError` for `--nsdf` because this
checkout does not contain matching pretrained disk/point or full mobile-arm geometry
checkpoints. There is no fallback geometry model.

Useful tuning flags:

```bash
uv run python examples/basic_demo.py --samples 256 --horizon 30 --steps 160 --seed 3 --headless
```

Run the compact safety sanity check without plotting:

```bash
uv run python examples/sanity_check.py --robot unicycle --algo brmppi --nsdf
uv run python examples/sanity_check.py --robot unicycle --algo brmppi --include-nsdf
```

This prints executed, sampled-rollout, and best-rollout clearance metrics for the analytic and neural-SDF barrier paths. Use it as the quick baseline when changing BR-MPPI cost terms.

The demo uses robot-specific default horizons so the BR-MPPI sample cloud visibly branches around obstacles. The single-integrator and dynamic-unicycle cases use 20-step rollout horizons, while unicycle and planar-quadrotor cases use 36-step rollout horizons by default. The mobile-arm default uses a 28-step horizon because its sampled footprint is much heavier to evaluate. All demos default to a 500-step maximum simulation cap and stop early on goal success. You can still override these with `--horizon`, `--samples`, `--steps`, and `--plot-samples`.

## Five-Method Visual Comparison

`test_compare_vis.py` runs one synchronized comparison using the committed tuned
configuration for each method. It intentionally includes only BR-MPPI, MPPI-CBF,
Shield-MPPI, SC-MPPI, and GS-MPPI; plain MPPI and penalty MPPI are not part of
this visualization. Select one of the five dynamics with a single flag:

```bash
uv run python test_compare_vis.py --dynamics unicycle
uv run python test_compare_vis.py --dynamics dynamic_unicycle
uv run python test_compare_vis.py --dynamics planar_quadrotor
uv run python test_compare_vis.py --dynamics single_integrator
uv run python test_compare_vis.py --dynamics mobile_arm
```

The GUI is the default. The four finalized non-mobile scenes reproduce the
obstacle layouts in the visual-comparison screenshots; the mobile-arm layout is
provisional. Twenty-four sampled rollouts are shown by default without changing
the tuned control parameters. Controller warm-up is excluded from the reported
per-command timing.

Save a synchronized video or final frame without opening a GUI:

```bash
uv run python test_compare_vis.py --dynamics unicycle --save-video --headless
uv run python test_compare_vis.py --dynamics unicycle --save-figure --headless
```

Default outputs are written under `output/compare_vis/`. Pass an explicit path
after `--save-video` or `--save-figure` to override it.

## Tuned Analytic/NSDF Benchmark

For a single BR-MPPI method, `random_benchmark.py` loads the repository's tuned
Optuna configuration by default. Omit `--nsdf` for the analytic h-function and add
it to change only the barrier source while retaining the same configuration,
random obstacle fields, and controller seeds:

```bash
uv run python examples/random_benchmark.py --robot unicycle --algo brmppi
uv run python examples/random_benchmark.py --robot unicycle --algo brmppi --nsdf
```

Use `--no-tuned-config` for the older generic benchmark defaults. NSDF output files
include `neural_sdf_tuned` in their names and record the selected checkpoint and
full tuned configuration. Single-integrator and mobile-arm NSDF benchmarks remain
unavailable until matching geometry checkpoints are trained. The completed 100-trial
analytic/NSDF comparison and optimization history are recorded in
[`docs/nsdf_benchmark_results.md`](docs/nsdf_benchmark_results.md).

The NSDF network weights, obstacle point clouds, and MLP queries use float32. Its
barrier Jacobian is computed directly from the network's local gradient and chained
through the robot-frame and lookahead transforms. The controller and analytic barrier
path remain x64, and the analytic barrier retains its existing finite-difference
Jacobian.

The default scene uses a larger random-looking circular obstacle field inspired by the BR-MPPI paper's branching-rollout illustration. A straight-line path from start to goal intersects obstacles for every robot, and the useful trajectories snake through clutter with obstacles on both sides instead of bypassing the field along open edges. The mobile-arm demo uses an even larger workspace with its own obstacle field because its fixed footprint is much larger. The green transparent lines are the MPPI sample cloud and the dashed blue line is the best sampled rollout at the current step.

The BR-MPPI implementation samples augmented controls `[u, alpha_dot]`, carries one class-K rate state per obstacle, projects each rollout control with the original closed-form weighted equality projection `A z = b`, and uses the mobile-arm nearest-barrier buffer cost `alpha_min / h_min`. The projection rows follow the original `mobile_arm` structure: `A = [dh/dx g(x), diag(h)]` and `b = -dh/dx f(x)`. Analytic h uses the existing finite-difference Jacobian; NSDF uses the direct MLP gradient plus the robot-frame/lookahead chain rule. Single-integrator and unicycle-style models use the immediate barrier state, while relative-degree-2 models use the same 5-step derived/lookahead barrier idea as `mobile_arm`: dynamic unicycle projects from a short future pose and planar quadrotor projects from `position + velocity * 5dt`. A small explicit projection margin tightens the obstacle barrier to absorb first-order discretization error. When a projected physical control reaches an actuator bound, the implementation recomputes the alpha components analytically so the bounded control still satisfies the BR equality rows. The BR rollout cost also adds a clearance shaping term and hard collision penalty so collided rollouts are strongly disfavored without changing the projection mechanism.

All analytic CBF/barrier baseline runs use the same h-function, projection margin,
finite-difference Jacobian, and relative-degree lookahead state as analytic BR-MPPI.
The `mppi_cbf` baseline solves the linearized CBF row
`dh/dx (f(x) + g(x)u) + alpha h(x) >= 0` with actuator bounds as a small
box-constrained QP in JAX. `shield_mppi`, `sc_mppi`, and `gs_mppi` intentionally do
not use that generic rollout QP path: Shield uses DCBF rollout penalties plus local
sequence repair, SC uses finite-horizon embedded-barrier-state feedback during
sampling, and GS uses a composite CBF with closed-form minimum-intervention safe
control. The `gs_cbf_residual_after_clipping` diagnostic reports any post-clipping
violation of the composite-CBF inequality.

Each run prints three safety diagnostics: executed trajectory clearance, minimum sampled-rollout clearance, and minimum best-rollout clearance. Collided samples can still appear in the MPPI population because sampling remains stochastic, but the selected best rollout should stay collision-free when the cost tuning is working.

If the executed trajectory reaches the goal, the simulation stops successfully. If it collides, the simulation stops at the first collision. Plots and videos draw a bold red `!` at the closest colliding body sample, and MP4 output holds that final collision frame briefly before ending.

The `mobile_arm` demo model uses a `1.5 x 1.0` planar base, two arm mounts at `[-0.5, 0]` and `[0.5, 0]`, two 4-link arms with link lengths `[0.8, 0.6, 0.4, 0.2]`, and a folded two-arm posture. The controller plans the two base controls plus eight joint velocity controls, while dense base and link edge samples are included in collision checking and visualization.

## SDF Notes

The analytic SDF in this demo is computed against circular obstacles. Each robot exposes representative body points:

- circular/point robots: center point with a body radius
- rectangle robots: rectangle corners plus center with a body-point radius
- mobile arm: base rectangle points plus sampled points along the two arm links

For each body point, the code computes `distance(point, obstacle_center) - obstacle_radius`, takes the closest obstacle, then takes the minimum across all robot body points and subtracts the robot body-point radius. This is a sampled footprint approximation for rectangles and arms, not an exact polygon-to-circle distance query.

When `--nsdf` is enabled for a supported robot, the barrier-rate h-function is evaluated by loading the matching pretrained Flax parameter dump from `sdf/trained_models`, sampling each circular obstacle boundary into a point cloud, transforming those points into the robot body frame, and taking the minimum predicted SDF value. The tuned rollout clearance shaping, collision rejection, and reported collision metric still use the analytic body-point clearance so those parts remain identical across analytic and neural runs. Consequently, this is a neural barrier-rate comparison, not a fully neural collision-checking pipeline. The analytic footprint also includes the robot's body-point padding, while the pretrained NSDF zero level set represents the nominal polygon.
