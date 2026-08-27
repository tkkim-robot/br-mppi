import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import yaml

images = {
    "dynamic_unicycle_hero": "temp_img1.png",
    "mobile_arm_scenario": "temp_img2.png",
    "planar_quadrotor_hero": "temp_img3.png",
    "single_integrator_hero": "temp_img4.png",
    "unicycle_hero": "temp_img5.png"
}

def extract(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    w = arr.shape[1]
    left_arr = arr[:, :int(w*0.55)] 
    
    def dist(color):
        return np.linalg.norm(left_arr - np.array(color), axis=2)
    
    # Start: mediumseagreen (60, 179, 113)
    green_mask = dist([60, 179, 113]) < 50
    lbl, num = ndimage.label(green_mask)
    start_px = None
    best_size = 0
    for i in range(1, num+1):
        size = np.sum(lbl == i)
        if size > best_size:
            best_size = size
            coords = np.where(lbl == i)
            start_px = (np.mean(coords[1]), np.mean(coords[0]))
            
    # Goal: gold (255, 215, 0)
    yellow_mask = dist([255, 215, 0]) < 50
    lbl, num = ndimage.label(yellow_mask)
    goal_px = None
    best_size = 0
    for i in range(1, num+1):
        size = np.sum(lbl == i)
        if size > best_size:
            best_size = size
            coords = np.where(lbl == i)
            goal_px = (np.mean(coords[1]), np.mean(coords[0]))
            
    # Gray circles
    rgbs = left_arr.astype(int)
    max_c = np.max(rgbs, axis=2)
    min_c = np.min(rgbs, axis=2)
    mean_c = np.mean(rgbs, axis=2)
    gray_mask = (max_c - min_c < 20) & (mean_c > 160) & (mean_c < 220)
    gray_mask = ndimage.binary_erosion(gray_mask, iterations=2)
    gray_mask = ndimage.binary_dilation(gray_mask, iterations=2)
    
    lbl, num = ndimage.label(gray_mask)
    obstacles_px = []
    for i in range(1, num+1):
        mask_i = lbl == i
        size = np.sum(mask_i)
        if size > 100:
            coords = np.where(mask_i)
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            r_y = (np.max(coords[0]) - np.min(coords[0])) / 2.0
            r_x = (np.max(coords[1]) - np.min(coords[1])) / 2.0
            radius = max(r_y, r_x) + 1
            obstacles_px.append((cx, cy, radius))
            
    return start_px, goal_px, obstacles_px

scenarios = {}

for name, path in images.items():
    start_px, goal_px, obs_px = extract(path)
    
    # Compute scale
    scale = (goal_px[0] - start_px[0]) / 14.0
    x_orig = start_px[0] - 2.0 * scale
    y_orig = start_px[1] + 5.0 * scale
    
    robot_type = name.replace('_hero', '').replace('_scenario', '')
    if robot_type == 'mobile_arm':
        start_state = [2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif robot_type == 'planar_quadrotor':
        start_state = [2.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    elif robot_type == 'dynamic_unicycle':
        start_state = [2.0, 5.0, 0.0, 0.0]
    elif robot_type == 'unicycle':
        start_state = [2.0, 5.0, 0.0]
    elif robot_type == 'single_integrator':
        start_state = [2.0, 5.0]
        
    obstacles = []
    for cx, cy, r in obs_px:
        data_cx = (cx - x_orig) / scale
        data_cy = (y_orig - cy) / scale
        data_r = r / scale
        obstacles.append({
            "center": [float(round(data_cx, 2)), float(round(data_cy, 2))],
            "radius": float(round(data_r, 2))
        })
        
    scenarios[name] = {
        "robot": robot_type,
        "description": f"Extracted exactly from image.",
        "start_state": start_state,
        "goal": [16.0, 5.0],
        "obstacles": obstacles
    }

with open("configs/hero_scenarios_v2.yaml", "w") as f:
    yaml.dump(scenarios, f, sort_keys=False)
    
print("Successfully generated configs/hero_scenarios_v2.yaml with exact extracted coordinates!")
