import yaml
import numpy as np
import subprocess
import json

def apply_smoothing(unicycle_margin, quad_margin):
    # Reload from raw extracted
    subprocess.run(["uv", "run", "python", "generate_extracted.py"], check=True, capture_output=True)
    subprocess.run(["uv", "run", "python", "fix_quad.py"], check=True, capture_output=True)
    
    with open("configs/hero_scenarios_v2.yaml", "r") as f:
        data = yaml.safe_load(f)
        
    def smooth_scenario(name, amplitude, margin):
        scenario = data[name]
        obstacles = scenario["obstacles"]
        new_obstacles = []
        xs = np.linspace(2.0, 16.0, 500)
        ys = 5.0 + amplitude * np.sin((xs - 2.0) / 14.0 * 2 * np.pi)
        
        for obs in obstacles:
            cx, cy = obs["center"]
            r = obs["radius"]
            dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
            min_idx = np.argmin(dists)
            min_d = dists[min_idx]
            px, py = xs[min_idx], ys[min_idx]
            
            req_dist = r + margin
            if min_d < req_dist:
                vec_x = cx - px
                vec_y = cy - py
                norm = np.sqrt(vec_x**2 + vec_y**2)
                if norm < 1e-4:
                    vec_x = 0.0
                    vec_y = 1.0 if amplitude > 0 else -1.0
                    norm = 1.0
                cx = px + (vec_x / norm) * req_dist
                cy = py + (vec_y / norm) * req_dist
                
            new_obstacles.append({
                "center": [float(round(cx, 2)), float(round(cy, 2))],
                "radius": r
            })
        data[name]["obstacles"] = new_obstacles

    smooth_scenario("dynamic_unicycle_hero", 2.0, unicycle_margin)
    smooth_scenario("planar_quadrotor_hero", -2.5, quad_margin)
    
    with open("configs/hero_scenarios_v2.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False)

margins = [(0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8)]
found_uni = False
found_quad = False

for u_m, q_m in margins:
    print(f"Testing margins: uni={u_m}, quad={q_m}")
    apply_smoothing(u_m, q_m)
    
    # Test Dynamic Unicycle
    print("  Running dynamic_unicycle...")
    cmd_uni = [
        "uv", "run", "python", "examples/hero_benchmark.py",
        "--scenario", "dynamic_unicycle_hero",
        "--robot-type", "dynamic_unicycle",
        "--algos", "brmppi", "mppi_cbf",
        "--headless", "--output-dir", "output/hero/tune"
    ]
    subprocess.run(cmd_uni, capture_output=True)
    
    # Check results
    import glob
    import os
    uni_files = glob.glob("output/hero/tune/dynamic_unicycle_hero_*.json")
    if uni_files:
        latest_uni = max(uni_files, key=os.path.getctime)
        with open(latest_uni) as f:
            res = json.load(f)
            br_res = next(r for r in res if r['algo'] == 'brmppi')
            cbf_res = next(r for r in res if r['algo'] == 'mppi_cbf')
            print(f"    BR-MPPI reached: {br_res['reached']}, CBF reached: {cbf_res['reached']}")
            if br_res['reached'] and not cbf_res['reached']:
                print(f"    *** Found good unicycle margin: {u_m}")
                
    # Test Quadrotor
    print("  Running planar_quadrotor...")
    cmd_quad = [
        "uv", "run", "python", "examples/hero_benchmark.py",
        "--scenario", "planar_quadrotor_hero",
        "--robot-type", "planar_quadrotor",
        "--algos", "brmppi", "mppi_cbf",
        "--headless", "--output-dir", "output/hero/tune"
    ]
    subprocess.run(cmd_quad, capture_output=True)
    quad_files = glob.glob("output/hero/tune/planar_quadrotor_hero_*.json")
    if quad_files:
        latest_quad = max(quad_files, key=os.path.getctime)
        with open(latest_quad) as f:
            res = json.load(f)
            br_res = next(r for r in res if r['algo'] == 'brmppi')
            cbf_res = next(r for r in res if r['algo'] == 'mppi_cbf')
            print(f"    BR-MPPI reached: {br_res['reached']}, CBF reached: {cbf_res['reached']}")
            if br_res['reached'] and not cbf_res['reached']:
                print(f"    *** Found good quad margin: {q_m}")

