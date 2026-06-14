# Single-Integrator Hyperparameter Tuning Results

Final benchmark summaries for the tuned single-integrator controller configs.

| dynamics | method | best trial | trials | reached | collisions | timeouts | success rate | collision rate | mean steps | mean command ms | worst clearance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_integrator | BR-MPPI | 12 | 100 | 95 | 0 | 5 | 0.950 | 0.000 | 218.4 | 26.39 | +0.000 |
| single_integrator | MPPI | 0 | 100 | 0 | 100 | 0 | 0.000 | 1.000 | 19.7 | 16.27 | -0.173 |
| single_integrator | Penalty MPPI | 12 | 100 | 86 | 0 | 14 | 0.860 | 0.000 | 286.5 | 16.61 | +0.000 |
| single_integrator | MPPI-CBF | 7 | 100 | 73 | 2 | 25 | 0.730 | 0.020 | 364.1 | 20.99 | -0.000 |
| single_integrator | Shield-MPPI | 14 | 100 | 80 | 13 | 7 | 0.800 | 0.130 | 257.0 | 25.07 | -0.149 |
| single_integrator | SC-MPPI | 2 | 100 | 2 | 0 | 98 | 0.020 | 0.000 | 792.1 | 106.75 | +0.395 |
| single_integrator | GS-MPPI | 10 | 100 | 78 | 22 | 0 | 0.780 | 0.220 | 159.4 | 21.08 | -0.095 |

The matching full configs are stored in `configs/tuned_hyperparameters.yaml`.

Note: SC-MPPI was intentionally stopped after 14 completed Optuna trials because each trial was much slower and performance remained poor. Its row and config come from the best completed 100-trial Optuna evaluation rather than a separate final re-evaluation pass.
