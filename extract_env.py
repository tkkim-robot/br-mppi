import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import sys
import json

def extract(img_path):
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)
    w = arr.shape[1]
    left_arr = arr[:, :int(w*0.55)] 
    
    def dist(color):
        return np.linalg.norm(left_arr - np.array(color), axis=2)
    
    # Old start color: mediumseagreen (60, 179, 113)
    d_green = dist([60, 179, 113])
    green_mask = d_green < 50
    lbl, num = ndimage.label(green_mask)
    start_px = None
    best_size = 0
    for i in range(1, num+1):
        size = np.sum(lbl == i)
        if size > best_size:
            best_size = size
            coords = np.where(lbl == i)
            start_px = (np.mean(coords[1]), np.mean(coords[0]))
            
    # Old goal color: gold (255, 215, 0)
    d_yellow = dist([255, 215, 0])
    yellow_mask = d_yellow < 50
    lbl, num = ndimage.label(yellow_mask)
    goal_px = None
    best_size = 0
    for i in range(1, num+1):
        size = np.sum(lbl == i)
        if size > best_size:
            best_size = size
            coords = np.where(lbl == i)
            goal_px = (np.mean(coords[1]), np.mean(coords[0]))
            
    # Old obstacles: slategray (112, 128, 144) with alpha=0.6 -> approx (198, 204, 211)
    # Let's just find anything greyish.
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
            radius = max(r_y, r_x)
            # compensate for erosion
            radius += 1
            obstacles_px.append((cx, cy, radius))
            
    return start_px, goal_px, len(obstacles_px), obstacles_px

if __name__ == "__main__":
    s, g, n, obs = extract(sys.argv[1])
    print(f"Start: {s}")
    print(f"Goal: {g}")
    print(f"Num Obs: {n}")
