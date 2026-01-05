#!/usr/bin/env python3
import os

# Correct data root relative to the current working directory
data_root = "datasets"

# List of KITTI drives you want to include
drives = [
    "2011_09_26_drive_0001_sync",
    "2011_09_26_drive_0002_sync",
    "2011_09_26_drive_0005_sync"
]

# Image folder inside each drive to use for monocular depth
image_folder = "image_02"

# Output file listing all images for training
output_file = "kitti_train_files.txt"

# Open output file
with open(output_file, "w") as f:
    for drive in drives:
        img_dir = os.path.join(data_root, drive, image_folder)
        
        # Check if folder exists
        if not os.path.exists(img_dir):
            print(f"Warning: Folder does not exist: {img_dir}")
            continue
        
        # List all PNG images in sorted order
        for file in sorted(os.listdir(img_dir)):
            if file.endswith(".png"):
                f.write(f"{drive}/{image_folder}/{file}\n")

print(f"{output_file} created with all images from drives!")

