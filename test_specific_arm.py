import yaml
import subprocess
import json
import glob
import os

with open("configs/hero_scenarios_v2.yaml", "r") as f:
    data = yaml.safe_load(f)

# The goal is BR-MPPI is the best, but others struggle.
obstacles = [
    # Top wall
    {"center": [4.0, 9.0], "radius": 1.5},
    {"center": [8.0, 8.5], "radius": 1.5},
    {"center": [12.0, 9.0], "radius": 1.5},
    
    # Bottom wall
    {"center": [4.0, 1.0], "radius": 1.5},
    {"center": [8.0, 1.5], "radius": 1.5},
    {"center": [12.0, 1.0], "radius": 1.5},
    
    # Zigzag islands forcing narrow passes (gap of 1.6m)
    {"center": [6.0, 6.5], "radius": 0.5},
    {"center": [6.0, 3.5], "radius": 0.5},
    
    {"center": [10.0, 6.5], "radius": 0.5},
    {"center": [10.0, 3.5], "radius": 0.5},
    
    # Extra debris to catch GS/Shield local planners (forces them into walls)
    {"center": [7.0, 4.0], "radius": 0.3},
    {"center": [11.0, 6.0], "radius": 0.3},
    
    # Shield tends to follow the path very closely but can't see far ahead
    # If we put a wall blocking the direct path that requires an early turn
    {"center": [13.2, 5.0], "radius": 0.52}, 
    {"center": [13.2, 5.8], "radius": 0.5},
    {"center": [13.2, 4.2], "radius": 0.5},
]

data["mobile_arm_scenario"]["obstacles"] = obstacles
data["mobile_arm_scenario"]["description"] = "A highly complex retraction challenge for the mobile arm."

with open("configs/hero_scenarios_v2.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("Testing tricky retraction environment...")
cmd = [
    "uv", "run", "python", "examples/hero_benchmark.py", 
    "--scenario", "mobile_arm_scenario", 
    "--robot-type", "mobile_arm", 
    "--algos", "brmppi", "shield_mppi", "gs_mppi", "sc_mppi", "mppi_cbf",
    "--headless", "--output-dir", "output/hero/tune_arm"
]
subprocess.run(cmd)

files = glob.glob("output/hero/tune_arm/mobile_arm_scenario_mobile_arm_results_*.json")
if files:
    latest = max(files, key=os.path.getctime)
    with open(latest) as f:
        res = json.load(f)
    print("RESULTS:")
    for r in res:
        print(f"  {r['algo']}: Reached={r['reached']}, Collision={r['collision']}, Steps={r['steps']}")
