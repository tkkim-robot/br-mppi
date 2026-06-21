# Dynamic-Unicycle Hyperparameter Tuning Results

Final benchmark summaries for the tuned dynamic-unicycle controller configs.

| dynamics | method | best trial | trials | reached | collisions | timeouts | deadlocks | success rate | collision rate | mean steps | mean command ms | worst clearance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dynamic_unicycle | BR-MPPI | 19 | 100 | 100 | 0 | 0 | 0 | 1.000 | 0.000 | 66.6 | 47.76 | +0.032 |
| dynamic_unicycle | MPPI | 0 | 100 | 0 | 100 | 0 | 0 | 0.000 | 1.000 | 15.3 | 35.33 | -0.373 |
| dynamic_unicycle | Penalty MPPI | 2 | 100 | 100 | 0 | 0 | 0 | 1.000 | 0.000 | 66.1 | 35.75 | +0.001 |
| dynamic_unicycle | MPPI-CBF | 8 | 100 | 72 | 17 | 11 | 7 | 0.720 | 0.170 | 168.9 | 36.74 | -0.064 |
| dynamic_unicycle | Shield-MPPI | 11 | 100 | 95 | 1 | 4 | 0 | 0.950 | 0.010 | 184.9 | 51.45 | -0.055 |
| dynamic_unicycle | SC-MPPI | 8 | 100 | 10 | 83 | 7 | 0 | 0.100 | 0.830 | 181.5 | 180.55 | -0.377 |
| dynamic_unicycle | GS-MPPI | 11 | 100 | 69 | 31 | 0 | 0 | 0.690 | 0.310 | 49.9 | 45.20 | -0.457 |

The matching full configs are stored in `configs/tuned_hyperparameters.yaml`.

The compact generated artifacts are stored under `output/tuning/`:

- `dynamic_unicycle_<algorithm>_best_config.json`
- `dynamic_unicycle_<algorithm>_study.json`
- `dynamic_unicycle_<algorithm>_best_benchmark.json`
- `dynamic_unicycle_<algorithm>_best_benchmark.md`

W&B runs:

| method | run |
|---|---|
| BR-MPPI | `5la7eo6z` |
| MPPI | `89j6oglf` |
| Penalty MPPI | `3k3llsxj` |
| MPPI-CBF | `ssjwqhta` |
| Shield-MPPI | `j4ottexd` |
| SC-MPPI | `15ohy7db, agn3ye16` |
| GS-MPPI | `wlmrvn5w` |

Notes:

- BR-MPPI and Penalty MPPI both reached 100/100 with zero collisions; BR-MPPI kept a larger worst clearance margin.
- SC-MPPI was resumed after a machine restart. The interrupted trial is marked `FAIL` in the study; the final report uses 10 completed trials.
- SC-MPPI remained both slow and collision-heavy for this dynamics.
