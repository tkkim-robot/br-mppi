#!/bin/bash
mkdir -p "output/hero/final_extracted_v2"

PAIRS=(
  "single_integrator:single_integrator_hero"
  "unicycle:unicycle_hero"
  "dynamic_unicycle:dynamic_unicycle_hero"
  "planar_quadrotor:planar_quadrotor_hero"
  "mobile_arm:mobile_arm_scenario"
)

for pair in "${PAIRS[@]}"; do
  robot="${pair%%:*}"
  scenario="${pair##*:}"

  echo "================================================="
  echo "Running Robot: $robot | Scenario: $scenario"
  
  uv run python examples/hero_benchmark.py \
    --scenario "$scenario" \
    --robot-type "$robot" \
    --algos brmppi mppi_cbf shield_mppi sc_mppi gs_mppi \
    --save-figure \
    --save-video \
    --headless \
    --output-dir "output/hero/final_extracted_v2"
    
  echo "Finished $robot!"
done

echo "ALL V2 EXTRACTED RENDERS COMPLETE!"
