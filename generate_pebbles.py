import random
random.seed(123)

yaml_str = """
dense_pebbles:
  robot: dynamic_unicycle
  description: "A highly complex field of 120 very small obstacles scattered like a minefield."
  start_state: [1.0, 5.0, 0.0, 0.0]
  goal: [18.0, 5.0]
  obstacles:"""

# Generate 120 tiny obstacles
for _ in range(120):
    x = round(random.uniform(2.5, 16.5), 2)
    y = round(random.uniform(0.5, 9.5), 2)
    r = round(random.uniform(0.08, 0.22), 2) # Very small radiuses!
    yaml_str += f"\n    - {{center: [{x}, {y}], radius: {r}}}"

with open("configs/hero_scenarios.yaml", "a") as f:
    f.write(yaml_str + "\n")

print("Generated dense_pebbles scenario in hero_scenarios.yaml!")
