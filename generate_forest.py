import random
random.seed(42)

yaml_str = """
mega_forest:
  robot: dynamic_unicycle
  description: "A massive, dense forest of 60 scattered obstacles resembling the benchmark plot."
  start_state: [1.0, 5.0, 0.0, 0.0]
  goal: [18.0, 5.0]
  obstacles:"""

for _ in range(60):
    x = round(random.uniform(3.0, 17.0), 2)
    y = round(random.uniform(1.0, 9.0), 2)
    r = round(random.uniform(0.15, 0.55), 2)
    yaml_str += f"\n    - {{center: [{x}, {y}], radius: {r}}}"

with open("configs/hero_scenarios.yaml", "a") as f:
    f.write(yaml_str + "\n")

print("Generated mega_forest in hero_scenarios.yaml!")
