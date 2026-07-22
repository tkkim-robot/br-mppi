# BR-MPPI Analytic vs Neural-SDF Benchmark

Previous NSDF run: 2026-07-21
Optimized NSDF run: 2026-07-22

## Methodology

No controller hyperparameters were retuned. For each supported dynamics model, the
analytic, previous-NSDF, and optimized-NSDF reports contain the exact same saved
Optuna `best_config`. The NSDF runs use 100 paired random obstacle fields (field seeds
13-112), controller seeds 7-106, 20 obstacles, a 900-step cap, and identical stopping
and deadlock settings.

Each trial performs a synchronized warm-up command and resets the controller before
starting either timer. Every measured command is also synchronized. Consequently,
`mean command ms` is steady-state inference latency: JAX compilation/startup is
excluded from both the command samples and trial wall time.

The optimized NSDF path changes only neural barrier evaluation:

- checkpoint weights, obstacle points, and MLP queries use float32;
- value-only queries no longer compute unused gradients;
- projection barriers and their direct MLP Jacobians are evaluated together, then
  chained through the robot-frame and relative-degree lookahead transforms;
- sampled obstacle points are cached per complete obstacle field; and
- the unused post-command BR constraint diagnostic is not recomputed for neural BR-MPPI.

The controller and analytic barrier path remain x64. The analytic path retains its
existing finite-difference Jacobian and post-command diagnostic. Existing analytic
results were not rerun or replaced; two-command analytic trace fingerprints for all
three supported dynamics were identical before and after the optimization.

## Checkpoint coverage

| dynamics | NSDF status | checkpoint / reason |
| --- | --- | --- |
| single integrator | unavailable | No matching disk/point checkpoint is present. |
| unicycle | supported | `link1_model_4_16.npy` (rectangle 1.0 x 0.4) |
| dynamic unicycle | supported | `link1_model_4_16.npy` (rectangle 1.0 x 0.4) |
| planar quadrotor | supported | `link7_model_4_16.npy` (rectangle 0.56 x 0.28) |
| mobile arm | unavailable | No checkpoint represents the composite base plus two articulated four-link arms. |

## Outcome and safety results

`R/C/T` means reached/collision/timeout over 100 trials. Deadlocks (`D`) are a subset
of timeouts. Worst clearance uses the common analytic body-point metric for every
source, so negative values denote a collision under the benchmark's shared metric.

| dynamics | barrier source | R/C/T | D | success | mean steps | mean final error | worst clearance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| single integrator | analytic | 95/0/5 | not recorded | 95% | 218.42 | 0.695 | +0.000002 |
| single integrator | NSDF | unavailable | — | — | — | — | — |
| unicycle | analytic | 87/0/13 | 5 | 87% | 226.54 | 1.431 | +0.000002 |
| unicycle | previous NSDF (x64, finite difference) | 84/8/8 | 0 | 84% | 238.01 | 1.655 | -0.025385 |
| unicycle | optimized NSDF (f32, direct Jacobian) | 84/11/5 | 0 | 84% | 220.86 | 1.685 | -0.047054 |
| dynamic unicycle | analytic | 100/0/0 | 0 | 100% | 66.60 | 0.449 | +0.031675 |
| dynamic unicycle | previous NSDF (x64, finite difference) | 100/0/0 | 0 | 100% | 66.08 | 0.451 | +0.008144 |
| dynamic unicycle | optimized NSDF (f32, direct Jacobian) | 100/0/0 | 0 | 100% | 66.35 | 0.449 | +0.022144 |
| planar quadrotor | analytic | 98/0/2 | 1 | 98% | 204.98 | 0.532 | +0.000392 |
| planar quadrotor | previous NSDF (x64, finite difference) | 93/1/6 | 0 | 93% | 230.31 | 1.023 | -0.006227 |
| planar quadrotor | optimized NSDF (f32, direct Jacobian) | 94/0/6 | 0 | 94% | 214.42 | 0.838 | +0.000004 |
| mobile arm | analytic | 95/0/5 | 0 | 95% | 293.20 | 1.126 | +0.003944 |
| mobile arm | NSDF | unavailable | — | — | — | — | — |

## Runtime results

| supported dynamics | analytic ms | previous NSDF ms | optimized NSDF ms | previous-to-optimized speedup | latency reduction | optimized / analytic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unicycle | 38.82 | 179.09 | 18.46 | 9.70x | 89.69% | 0.48x |
| dynamic unicycle | 47.76 | 204.32 | 18.97 | 10.77x | 90.71% | 0.40x |
| planar quadrotor | 52.23 | 214.80 | 23.48 | 9.15x | 89.07% | 0.45x |

The optimized aggregate success rates are effectively unchanged from the previous
NSDF runs: 84% to 84% for unicycle, 100% to 100% for dynamic unicycle, and 93% to
94% for planar quadrotor. Paired categorical outcomes remained the same on 87, 100,
and 97 of the 100 seeds, respectively. Unicycle's aggregate success is unchanged but
its failure composition moved from 8 collisions/8 timeouts to 11 collisions/5
timeouts, so the speedup should not be interpreted as exact trajectory equivalence.

## Interpretation and caveats

The previous NSDF slowdown was an implementation cost, not a JIT warm-up artifact.
Finite-differencing the neural barrier multiplied MLP evaluations by state dimension,
and value-only paths also computed gradients that were discarded. Float32 neural
inference plus the direct Jacobian removes most of that work, reducing steady-state
latency by 89-91% while preserving aggregate success within one percentage point.

The optimized/analytic timing ratio is not a symmetric h-function microbenchmark.
The optimized neural path omits a post-command constraint diagnostic that BR-MPPI
does not consume, while the analytic path deliberately retains it so the existing
analytic baseline has no timing or behavioral side effect. The previous-to-optimized
NSDF speedup is therefore the primary performance comparison.

`--nsdf` replaces the BR-MPPI barrier-rate h evaluation. Tuned rollout clearance
shaping, rollout collision rejection, and reported collision checks remain analytic,
so this is not a fully neural collision-checking pipeline. The analytic footprint also
subtracts body-point padding (0.08 for the unicycles and 0.06 for planar quadrotor),
whereas each pretrained NSDF zero level set represents the nominal polygon without
that padding. Float32 arithmetic, direct rather than forward finite-difference
gradients, and nearest-point ties can change individual sampled trajectories.

Single integrator and mobile arm remain unavailable rather than failed: this checkout
does not contain checkpoints for their geometries.

## Raw artifacts

Previous NSDF reports:

- `output/benchmarks/nsdf/unicycle_brmppi_neural_sdf_tuned_100.json`
- `output/benchmarks/nsdf/dynamic_unicycle_brmppi_neural_sdf_tuned_100.json`
- `output/benchmarks/nsdf/planar_quadrotor_brmppi_neural_sdf_tuned_100.json`

Optimized NSDF reports:

- `output/benchmarks/nsdf_optimized/unicycle_brmppi_neural_sdf_float32_direct_tuned_100.json`
- `output/benchmarks/nsdf_optimized/dynamic_unicycle_brmppi_neural_sdf_float32_direct_tuned_100.json`
- `output/benchmarks/nsdf_optimized/planar_quadrotor_brmppi_neural_sdf_float32_direct_tuned_100.json`

The repository ignores `output/`; this tracked document is the durable result summary.
