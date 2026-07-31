# BR-MPPI Analytic vs Neural-SDF Benchmark

## Retrained-checkpoint benchmark (2026-07-30)

### Methodology

No controller hyperparameters were retuned. Each neural run loads the same saved
Optuna `best_config` as its analytic BR-MPPI reference and changes the barrier
h-function implementation only. The 100 trials use field seeds 13-112, controller
seeds 7-106, the same robot-specific obstacle count and step cap, and the same
stopping/deadlock rules.

Every neural trial performs a synchronized warm-up command, resets the controller,
and then synchronizes every measured command. JIT compilation and warm-up are
excluded from `mean command ms` and trial p95. The analytic reports are the existing
baseline artifacts and were not rerun or replaced. Those older files preserve trial
and field seeds but predate explicit controller-seed/runtime metadata, so the paired
report correctly marks artifact-level controller-seed and timing-policy validation
as unavailable.

The rigid-body neural rows use the retrained `link1` or `link7` model with float32
MLP evaluation and a direct Jacobian. The mobile-arm row is explicitly a guarded
analytic/neural hybrid: exact continuous rectangles select and upper-guard the
learned base/link narrow phase. It is not a pure learned SDF evaluation.

### All analytic baseline outcomes

`R/C/T` means reached/collision/timeout over 100 trials. Deadlocks are included in
timeouts. These are the finalized analytic h-function results for every method and
dynamics model.

| dynamics | BR-MPPI | MPPI | Penalty MPPI | MPPI-CBF | Shield MPPI | SC-MPPI | GS-MPPI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single integrator | 95/0/5 | 0/100/0 | 86/0/14 | 73/2/25 | 80/13/7 | 2/0/98 | 78/22/0 |
| unicycle | 87/0/13 | 0/100/0 | 87/5/8 | 12/86/2 | 64/6/30 | 17/1/82 | 64/36/0 |
| dynamic unicycle | 100/0/0 | 0/100/0 | 100/0/0 | 72/17/11 | 95/1/4 | 10/83/7 | 69/31/0 |
| planar quadrotor | 98/0/2 | 0/100/0 | 93/0/7 | 64/21/15 | 75/4/21 | 12/3/85 | 74/14/12 |
| mobile arm | 95/0/5 | 0/100/0 | 95/4/1 | 3/97/0 | 89/10/1 | 17/41/42 | 49/51/0 |

### BR-MPPI analytic vs retrained NSDF outcomes

Worst clearance uses the shared historical analytic body-point metric. A negative
value would denote collision. `D` is the number of deadlock timeouts.

| dynamics | barrier h | R/C/T | D | success | mean steps | mean final error | worst clearance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unicycle | analytic | 87/0/13 | 5 | 87% | 226.54 | 1.431 | +0.000002 |
| unicycle | retrained NSDF | 81/0/19 | 11 | 81% | 238.61 | 2.129 | +0.000792 |
| dynamic unicycle | analytic | 100/0/0 | 0 | 100% | 66.60 | 0.449 | +0.031675 |
| dynamic unicycle | retrained NSDF | 97/0/3 | 3 | 97% | 76.24 | 0.449 | +0.106995 |
| planar quadrotor | analytic | 98/0/2 | 1 | 98% | 204.98 | 0.532 | +0.000392 |
| planar quadrotor | retrained NSDF | 96/0/4 | 3 | 96% | 224.69 | 0.790 | +0.001990 |
| mobile arm | analytic | 95/0/5 | 0 | 95% | 293.20 | 1.126 | +0.003944 |
| mobile arm | guarded NSDF hybrid | 93/0/7 | 0 | 93% | 333.24 | 1.268 | +0.004507 |

The mobile neural run also reports zero continuous-rectangle collisions and a
worst continuous clearance of `+0.004425 m`. The historical analytic report did
not record this newer metric, so no paired continuous-geometry delta is claimed.

### Runtime and paired changes

| dynamics | analytic mean ms | neural mean ms | analytic p95 ms | neural p95 ms | neural / analytic | speedup | success delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unicycle | 38.82 | 19.00 | 40.32 | 20.06 | 0.490x | 2.04x | -6 pp |
| dynamic unicycle | 47.76 | 19.54 | 49.16 | 20.98 | 0.409x | 2.44x | -3 pp |
| planar quadrotor | 52.23 | 23.20 | 53.95 | 24.21 | 0.444x | 2.25x | -2 pp |
| mobile arm | 145.27 | 46.64 | 152.29 | 52.80 | 0.321x | 3.12x | -2 pp |

| dynamics | neural success W/L/T vs analytic | exact outcome agreement | collisions introduced/resolved |
| --- | ---: | ---: | ---: |
| unicycle | 4/10/86 | 84/100 | 0/0 |
| dynamic unicycle | 0/3/97 | 97/100 | 0/0 |
| planar quadrotor | 1/3/96 | 96/100 | 0/0 |
| mobile arm | 1/3/96 | 96/100 | 0/0 |

The retrained h-functions introduce no executed collisions in any of the 400 paired
trials. They reduce steady-state controller time by 51-68%, but success falls by
2-6 percentage points because additional trials terminate as timeouts/deadlocks.
Better pointwise SDF accuracy therefore does not imply identical closed-loop
behavior: the empirical model margin, float32/direct gradients, nearest-sample ties,
and the changed h surface all alter BR-MPPI projection and sampling.

The single integrator remains unsupported rather than failed. Its footprint is a
disk/point model, while the implemented trainer and supplied checkpoints represent
rectangles. Selecting `--nsdf` for it raises `NotImplementedError` rather than
silently changing the robot geometry.
