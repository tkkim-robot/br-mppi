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

Choose the controller:

```bash
uv run python examples/basic_demo.py --algo brmppi
uv run python examples/basic_demo.py --algo mppi
uv run python examples/basic_demo.py --algo penalty_mppi
```

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

Pretrained NSDF support is enabled only when a vendored `sdf/trained_models`
checkpoint matches the robot footprint used here:

- `unicycle`: `link1_model_4_16.npy`, rectangle `1.0 x 0.4`
- `dynamic_unicycle`: `link1_model_4_16.npy`, rectangle `1.0 x 0.4`
- `planar_quadrotor`: `link7_model_4_16.npy`, quad2 `0.56 x 0.28`

`single_integrator` and `mobile_arm` raise `NotImplementedError` for `--nsdf` because this
checkout does not contain matching pretrained disk/point or full mobile-arm geometry
checkpoints. The previous online/random-feature SDF fitter has been removed from the runnable path.

Useful tuning flags:

```bash
uv run python examples/basic_demo.py --samples 256 --horizon 30 --steps 160 --seed 3 --headless
```

The default scene uses a larger random-looking circular obstacle field inspired by the BR-MPPI paper's branching-rollout illustration. A straight-line path from start to goal intersects obstacles for every robot, and the useful trajectories snake through clutter with obstacles on both sides instead of bypassing the field along open edges. The mobile-arm demo uses an even larger workspace with its own obstacle field because its fixed footprint is much larger. BR-MPPI is configured to reach the goal without collision in these scenes while still allowing baseline MPPI variants for comparison. The green transparent lines are a fixed-size MPPI sample cloud (`--plot-samples`, default `100`) and the dashed blue line is the best sampled rollout at the current step.

If a rollout collides, the simulation stops at the first collision. Plots and videos draw a bold red `!` at the closest colliding body sample, and MP4 output holds that final collision frame briefly before ending.

The `mobile_arm` demo model uses a `1.5 x 1.0` planar base, two arm mounts at `[-0.5, 0]` and `[0.5, 0]`, fixed 4-link arms with link lengths `[0.8, 0.6, 0.4, 0.2]`, and a folded two-arm posture. The controller currently plans the mobile base motion while the base and fixed arm links are included in collision checking and visualization.

## SDF Notes

The analytic SDF in this demo is computed against circular obstacles. Each robot exposes representative body points:

- circular/point robots: center point with a body radius
- rectangle robots: rectangle corners plus center with a body-point radius
- mobile arm: base rectangle points plus sampled points along the two arm links

For each body point, the code computes `distance(point, obstacle_center) - obstacle_radius`, takes the closest obstacle, then takes the minimum across all robot body points and subtracts the robot body-point radius. This is a sampled footprint approximation for rectangles and arms, not an exact polygon-to-circle distance query.

When `--nsdf` is enabled for a supported robot, the barrier-rate h-function is evaluated by loading the matching pretrained Flax parameter dump from `sdf/trained_models`, sampling each circular obstacle boundary into a point cloud, transforming those points into the robot body frame, and taking the minimum predicted SDF value. Rollout collision checks still use the analytic body-point clearance so the reported collision metric is consistent across analytic and neural runs.
