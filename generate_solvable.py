import random
import math

random.seed(99)

yaml_str = """
winding_corridor:
  robot: dynamic_unicycle
  description: "A dense field with a mathematically guaranteed S-curve safe corridor."
  start_state: [1.0, 5.0, 0.0, 0.0]
  goal: [18.0, 5.0]
  obstacles:"""

# The safe path is a sine wave
def path_y(x):
    return 5.0 + 3.0 * math.sin((x - 1.0) / 17.0 * 2 * math.pi)

obstacles = []
attempts = 0

# Keep guessing random obstacles until we get 65 good ones
while len(obstacles) < 65 and attempts < 3000:
    x = round(random.uniform(2.0, 17.0), 2)
    y = round(random.uniform(0.5, 9.5), 2)
    r = round(random.uniform(0.3, 0.6), 2) # Medium sizes
    
    # Check how close this obstacle is to our invisible safe path
    min_dist = 999
    for i in range(101):
        px = 1.0 + i * (17.0 / 100.0)
        py = path_y(px)
        dist = math.hypot(x - px, y - py)
        if dist < min_dist:
            min_dist = dist
            
    # If the obstacle is at least (Radius + 0.8 meters) away from the curve, keep it!
    # This guarantees a corridor of at least 1.6 meters wide for the car to drive through.
    if min_dist > (r + 0.8): 
        obstacles.append((x, y, r))
        
    attempts += 1

for x, y, r in obstacles:
    yaml_str += f"\n    - {{center: [{x}, {y}], radius: {r}}}"

with open("configs/hero_scenarios.yaml", "a") as f:
    f.write(yaml_str + "\n")

print("Generated winding_corridor successfully!")
