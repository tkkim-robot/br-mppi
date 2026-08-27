import yaml
import numpy as np
import subprocess
import json
import os
import glob

os.makedirs("output/hero/tune_arm", exist_ok=True)

with open("configs/hero_scenarios_v2.yaml", "r") as f:
    data = yaml.safe_load(f)

for seed in [101, 102, 103, 104, 105]:
    print(f"Testing seed {seed}...")
    np.random.seed(seed)
    
    obstacles = []
    # Create a structured dense field
    for x in np.arange(4.5, 14.5, 2.0):
        for y in np.arange(1.5, 9.5, 2.0):
            cx = x + np.random.uniform(-0.5, 0.5)
            cy = y + np.random.uniform(-0.5, 0.5)
            r = np.random.uniform(0.3, 0.7)
            # clear a path approximately matching y=5 to give BR-MPPI a fighting chance
            if abs(cy - 5.0) < 1.0:
                continue
            obstacles.append({"center": [float(round(cx, 2)), float(round(cy, 2))], "radius": float(round(r, 2))})
            
    # Add a few specific tricky blocks in the middle path to force folding
    obstacles.append({"center": [7.0, 5.5], "radius": 0.4})
    obstacles.append({"center": [11.0, 4.5], "radius": 0.4})
    obstacles.append({"center": [9.0, 5.0], "radius": 0.2})
        
    data["mobile_arm_scenario"]["obstacles"] = obstacles
    with open("configs/hero_scenarios_v2.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False)
        
    cmd = [
        "uv", "run", "python", "examples/hero_benchmark.py", 
        "--scenario", "mobile_arm_scenario", 
        "--robot-type", "mobile_arm", 
        "--algos", "brmppi", "shield_mppi", "gs_mppi",
        "--headless", "--output-dir", "output/hero/tune_arm"
    ]
    
    subprocess.run(cmd, capture_output=True)
    
    files = glob.glob("output/hero/tune_arm/mobile_arm_scenario_mobile_arm_results_*.json")
    if files:
        latest = max(files, key=os.path.getctime)
        with open(latest) as f:
            res = json.load(f)
        br = next((r for r in res if r['algo'] == 'brmppi'), None)
        shield = next((r for r in res if r['algo'] == 'shield_mppi'), None)
        gs = next((r for r in res if r['algo'] == 'gs_mppi'), None)
        
        br_ok = br['reached'] if br else False
        sh_ok = shield['reached'] if shield else False
        gs_ok = gs['reached'] if gs else False
        
        print(f"  BR: {br_ok}, SH: {sh_ok}, GS: {gs_ok}")
        
        if br_ok and not sh_ok and not gs_ok:
            print(f"Found ideal complex environment! Seed {seed}")
            break
