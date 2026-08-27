import subprocess
import glob
import os

print("Re-rendering arm with correct configuration...")
cmd = [
    "uv", "run", "python", "examples/hero_benchmark.py", 
    "--scenario", "mobile_arm_scenario", 
    "--robot-type", "mobile_arm", 
    "--algos", "brmppi", "mppi_cbf", "shield_mppi", "sc_mppi", "gs_mppi", 
    "--save-video", "--save-figure",
    "--headless", "--output-dir", "output/hero/final_extracted_v2"
]
subprocess.run(cmd)

