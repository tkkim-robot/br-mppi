import yaml
import numpy as np
import subprocess

subprocess.run(["uv", "run", "python", "generate_extracted.py"], check=True, capture_output=True)
subprocess.run(["uv", "run", "python", "fix_quad.py"], check=True, capture_output=True)

yaml_path = "configs/hero_scenarios_v2.yaml"
with open(yaml_path, "r") as f:
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

# Use 0.7m margin for a nice smooth channel that is just narrow enough
if "dynamic_unicycle_hero" in data:
    smooth_scenario("dynamic_unicycle_hero", 2.0, 0.7)
    
if "planar_quadrotor_hero" in data:
    smooth_scenario("planar_quadrotor_hero", -2.5, 0.7)

with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("Applied final smooth configurations!")
