import cv2
import numpy as np
import argparse
import os
import glob
from PIL import Image

def find_latest_files(folder_path):
    files = glob.glob(os.path.join(folder_path, "envmap_env_*_*.png"))
    
    iterations = {}
    for file in files:
        basename = os.path.basename(file)
        iter_num = basename.split('_')[-1].replace('.png', '')
        
        if iter_num not in iterations:
            iterations[iter_num] = {}
            
        if 'cood1' in basename:
            iterations[iter_num]['cood1'] = file
        elif 'usage' in basename:
            iterations[iter_num]['usage'] = file
    
    latest_iter = max(iterations.keys())
    return iterations[latest_iter]['cood1'], iterations[latest_iter]['usage'], latest_iter

def mask_and_crop(image_path, mask_path):
    mask_img = Image.open(mask_path).convert('L')
    target_img = Image.open(image_path).convert('RGBA')
    
    mask_array = np.array(mask_img) / 255.0
    smoothed_mask = cv2.GaussianBlur(mask_array, (7, 7), 2)
    alpha = np.where(smoothed_mask > 0.6, 255, 0).astype(np.uint8)
    
    target_array = np.array(target_img)
    target_array[..., 3] = alpha
    
    coords = np.where(alpha > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    return Image.fromarray(target_array[y_min:y_max+1, x_min:x_max+1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='folder path containing envmap files')
    args = parser.parse_args()
    
    img_path, mask_path, iteration = find_latest_files(args.folder)
    folder_name = os.path.basename(args.folder.rstrip('/'))
    output_name = f"{folder_name}_masked_cropped_{iteration}.png"
    
    result = mask_and_crop(img_path, mask_path)
    result.save(output_name)
    print(f"processed iteration {iteration} from {folder_name}")
    print(f"saved result to {output_name}")