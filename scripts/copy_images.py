import os
import shutil

# Paths
images_list_path = 'example_dataset/images_to_process.txt'
destination_dir = 'example_dataset/images'

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Read image paths from file
with open(images_list_path, 'r') as f:
    image_paths = [line.strip() for line in f if line.strip()]

# Copy images
for img_path in image_paths:
    if os.path.isfile(img_path):
        shutil.copy(img_path, destination_dir)
    else:
        print(f"Warning: File not found - {img_path}")