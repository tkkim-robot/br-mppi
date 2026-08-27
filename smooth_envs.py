import yaml
import numpy as np

yaml_path = "configs/hero_scenarios_v2.yaml"
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

def smooth_scenario(name, amplitude, margin):
    scenario = data[name]
    obstacles = scenario["obstacles"]
    new_obstacles = []
    
    # Define a smooth S-curve path
    xs = np.linspace(2.0, 16.0, 500)
    ys = 5.0 + amplitude * np.sin((xs - 2.0) / 14.0 * 2 * np.pi)
    
    for obs in obstacles:
        cx, cy = obs["center"]
        r = obs["radius"]
        
        # Find closest point on the path
        dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
        min_idx = np.argmin(dists)
        min_d = dists[min_idx]
        px, py = xs[min_idx], ys[min_idx]
        
        req_dist = r + margin
        if min_d < req_dist:
            # Push the obstacle outward along the vector from the path
            vec_x = cx - px
            vec_y = cy - py
            norm = np.sqrt(vec_x**2 + vec_y**2)
            
            if norm < 1e-4:
                # If perfectly on the path, push vertically based on amplitude direction
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

# Provide a 1.2m safety margin (2.4m wide corridor) for high speed vehicles
if "dynamic_unicycle_hero" in data:
    smooth_scenario("dynamic_unicycle_hero", 2.5, 1.4)
    
if "planar_quadrotor_hero" in data:
    smooth_scenario("planar_quadrotor_hero", -2.5, 1.4)

with open(yaml_path, "w") as f:
    yaml.dump(data, f, sort_keys=False)

print("Smoothed dynamic unicycle and planar quadrotor corridors!")
