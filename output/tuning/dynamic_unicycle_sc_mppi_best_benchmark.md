# Best Benchmark: dynamic_unicycle / sc_mppi

- Study: `dynamic_unicycle_sc_mppi`
- Best trial: `8`
- Best objective: `0.095932`
- Benchmark trials: `100`
- JSON: `output/tuning/dynamic_unicycle_sc_mppi_best_benchmark.json`

| dynamics | method | trials | reached | collisions | timeouts | deadlocks | success rate | collision rate | mean reach/trial steps | mean command ms | worst clearance |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dynamic_unicycle | sc_mppi | 100 | 10 | 83 | 7 | 0 | 0.100 | 0.830 | 181.5 | 180.55 | -0.377 |

## Best Config

```json
{
  "horizon": 36,
  "samples": 240,
  "dt": 0.1,
  "temperature": 0.5,
  "noise_scale": 0.6500000000000001,
  "goal_weight": 0.25,
  "final_goal_weight": 10.0,
  "control_weight": 0.08,
  "safety_weight": 18.0,
  "collision_weight": 10000.0,
  "barrier_alpha": 0.2,
  "alpha_rate_bound": 13.0,
  "alpha_state_bound": 5.0,
  "alpha_noise_scale": 1.5,
  "alpha_projection_inverse_weight": 0.2,
  "bound_penalty": 0.1,
  "barrier_buffer_distance": 0.05,
  "barrier_projection_margin": 0.13999999999999999,
  "barrier_alpha_cost_weight": 0.0,
  "br_clearance_margin": 0.5500000000000002,
  "br_clearance_weight": 50.0,
  "br_collision_weight": 100000.0,
  "cbf_alpha": 1.0,
  "cbf_qp_iterations": 35,
  "cbf_qp_rho": 8.0,
  "cbf_qp_regularization": 1e-08,
  "shield_cbf_penalty_weight": 900.0,
  "shield_alpha": 0.98,
  "shield_repair_horizon": 6,
  "shield_repair_steps": 8,
  "shield_repair_step_size": 0.04,
  "sc_barrier_eps": 0.001,
  "sc_barrier_gamma": 0.75,
  "sc_feedback_gain": 0.08,
  "sc_feedback_clip": 1.2,
  "sc_feedback_iterations": 3,
  "sc_feedback_regularization": 0.01,
  "sc_beta_cost_weight": 1.0,
  "sc_state_cost_weight": 0.02,
  "sc_control_cost_weight": 0.08,
  "sc_use_combined_barrier": true,
  "gs_softmin_rho": 25.0,
  "gs_closed_form_gamma": 2.0,
  "gs_alpha": 1.0,
  "gs_composite_margin": 0.0,
  "plot_samples": 0
}
```
