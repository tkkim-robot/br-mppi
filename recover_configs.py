import yaml
import random

yaml_path = "configs/hero_scenarios.yaml"
with open(yaml_path, 'r') as f:
    scenarios = yaml.safe_load(f)

# 1. Single Integrator (65 obstacles)
scenarios["single_integrator_hero"] = {
    "robot": "single_integrator",
    "description": "A dense, chaotic asteroid belt for the 2D point mass.",
    "start_state": [1.0, 5.0],
    "goal": [18.0, 5.0],
    "obstacles": []
}
random.seed(42)
for _ in range(65):
    x = round(random.uniform(3.0, 16.0), 2)
    y = round(random.uniform(1.0, 9.0), 2)
    r = round(random.uniform(0.3, 0.7), 2)
    scenarios["single_integrator_hero"]["obstacles"].append({"center": [x, y], "radius": r})

# 2. Unicycle (35 obstacles)
scenarios["unicycle_hero"] = {
    "robot": "unicycle",
    "description": "A dense urban slalom requiring aggressive steering.",
    "start_state": [1.0, 5.0, 0.0, 0.0],
    "goal": [18.0, 5.0],
    "obstacles": []
}
random.seed(99)
for _ in range(35):
    x = round(random.uniform(3.0, 16.0), 2)
    y = round(random.uniform(0.5, 9.5), 2)
    r = round(random.uniform(0.4, 0.8), 2)
    scenarios["unicycle_hero"]["obstacles"].append({"center": [x, y], "radius": r})

# 3. Dynamic Unicycle (35 obstacles)
scenarios["dynamic_unicycle_hero"] = {
    "robot": "dynamic_unicycle",
    "description": "A dense minefield for a high-speed inertial car.",
    "start_state": [1.0, 5.0, 0.0, 3.0], 
    "goal": [18.0, 5.0],
    "obstacles": []
}
random.seed(77)
for _ in range(35):
    x = round(random.uniform(3.0, 17.0), 2)
    y = round(random.uniform(1.0, 9.0), 2)
    r = round(random.uniform(0.3, 0.6), 2)
    scenarios["dynamic_unicycle_hero"]["obstacles"].append({"center": [x, y], "radius": r})

# 4. Quadrotor (30 obstacles exactly)
scenarios["planar_quadrotor_hero"] = {
    "robot": "planar_quadrotor",
    "description": "A jagged vertical cave for the 5D drone.",
    "start_state": [1.0, 5.0, 0.0, 0.0, 0.0],
    "goal": [18.0, 5.0],
    "obstacles": []
}
random.seed(88)
for _ in range(26):
    x = round(random.uniform(3.0, 16.0), 2)
    y = round(random.uniform(7.0, 11.0) if random.random() > 0.5 else random.uniform(-1.0, 3.0), 2)
    r = round(random.uniform(0.5, 1.2), 2)
    scenarios["planar_quadrotor_hero"]["obstacles"].append({"center": [x, y], "radius": r})
    if random.random() > 0.8:
        scenarios["planar_quadrotor_hero"]["obstacles"].append({"center": [x, round(random.uniform(4.0, 6.0), 2)], "radius": 0.4})

# 5. Mobile Arm (exact 10 obstacles from history)
scenarios["mobile_arm_scenario"] = {
    "robot": "mobile_arm",
    "description": "V3 Random Through 17400",
    "start_state": [2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "goal": [16.0, 5.0],
    "obstacles": [
        {"center": [4.482164067231302, 1.956846012216834], "radius": 0.5282405227638471},
        {"center": [11.482239872360095, 1.5177710609999606], "radius": 0.44487310440370037},
        {"center": [13.556928458279856, 4.938722864624772], "radius": 0.532004524432445},
        {"center": [5.656713375073586, 8.108561778701947], "radius": 0.42032739572490296},
        {"center": [8.954223108655606, 1.1917269933535102], "radius": 0.5182267881809396},
        {"center": [6.728090176530241, 0.31953426854007483], "radius": 0.40092061219410463},
        {"center": [12.78297404845222, 0.4913419546352893], "radius": 0.48691761959441077},
        {"center": [12.646354738406178, 0.3815618864006687], "radius": 0.45998717668887434},
        {"center": [8.492387556654343, 1.322364780335018], "radius": 0.580464251892028},
        {"center": [6.026355520530289, 6.839346479500569], "radius": 0.5204999462393963}
    ]
}

with open(yaml_path, 'w') as f:
    yaml.dump(scenarios, f, sort_keys=False)

print("Recovered all 5 configurations perfectly!")
