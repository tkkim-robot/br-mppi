import yaml

with open("configs/hero_scenarios_v2.yaml", "r") as f:
    data = yaml.safe_load(f)

if "planar_quadrotor_hero" in data:
    # Fix the state dimension to 5 elements instead of 6
    data["planar_quadrotor_hero"]["start_state"] = [2.0, 5.0, 0.0, 0.0, 0.0]

with open("configs/hero_scenarios_v2.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)
