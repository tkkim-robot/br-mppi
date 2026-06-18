# Unicycle Hyperparameter Tuning Results

Final benchmark summaries for the tuned unicycle controller configs.

| dynamics | method | best trial | trials | reached | collisions | timeouts | deadlocks | success rate | collision rate | mean steps | mean command ms | worst clearance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unicycle | BR-MPPI | 18 | 100 | 87 | 0 | 13 | 5 | 0.870 | 0.000 | 226.5 | 38.82 | +0.000 |
| unicycle | MPPI | 0 | 100 | 0 | 100 | 0 | 0 | 0.000 | 1.000 | 22.8 | 27.75 | -0.172 |
| unicycle | Penalty MPPI | 5 | 100 | 87 | 5 | 8 | 0 | 0.870 | 0.050 | 254.0 | 27.93 | -0.054 |
| unicycle | MPPI-CBF | 9 | 100 | 12 | 86 | 2 | 2 | 0.120 | 0.860 | 95.1 | 31.33 | -0.129 |
| unicycle | Shield-MPPI | 13 | 100 | 64 | 6 | 30 | 4 | 0.640 | 0.060 | 406.2 | 38.75 | -0.075 |
| unicycle | SC-MPPI | 0 | 100 | 17 | 1 | 82 | 0 | 0.170 | 0.010 | 849.2 | 92.14 | -0.121 |
| unicycle | GS-MPPI | 0 | 100 | 64 | 36 | 0 | 0 | 0.640 | 0.360 | 129.3 | 34.65 | -0.096 |

The matching full configs are stored in `configs/tuned_hyperparameters.yaml`.

The compact generated artifacts are stored under `output/tuning/`:

- `unicycle_<algorithm>_best_config.json`
- `unicycle_<algorithm>_study.json`
- `unicycle_<algorithm>_best_benchmark.json`
- `unicycle_<algorithm>_best_benchmark.md`

W&B runs:

| method | run |
|---|---|
| BR-MPPI | `mcc7up9x` |
| MPPI | `ep6ttzfw` |
| Penalty MPPI | `yaifpw7o` |
| MPPI-CBF | `3fqwl9ve` |
| Shield-MPPI | `ceytqnlj` |
| SC-MPPI | `b9htj27x` |
| GS-MPPI | `9m9bzot8` |

Notes:

- BR-MPPI and Penalty MPPI tied on success rate, but BR-MPPI had zero collisions in the final benchmark.
- SC-MPPI was capped at 10 Optuna trials because each trial was substantially slower than the other baselines and performance remained poor.
- The BR-MPPI failure snapshots are stored in `output/diagnostics/unicycle_brmppi_failures/`.
