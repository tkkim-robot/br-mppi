import yaml
import copy

with open("configs/hero_scenarios.yaml", "r") as f:
    data = yaml.safe_load(f)

new_data = {}
for k, v in data.items():
    new_data[k] = v

with open("configs/hero_scenarios_v2.yaml", "w") as f:
    yaml.dump(new_data, f, sort_keys=False)
