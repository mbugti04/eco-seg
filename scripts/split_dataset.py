# takes in dataset folder
# copies half of dataset/images into dataset_half_1 and other half into dataset_half_2
# copies half of dataset/labels into dataset_half_1 and other half into dataset_half_2

import os
import glob
import shutil
import numpy as np

dataset_name = "dataset 10 dec 25"

directory = f"dataset 10 dec 25"

images_path = f"{directory}/images"
labels_path = f"{directory}/labels"


half_1_directory = f"{dataset_name}_half_1"
half_2_directory = f"{dataset_name}_half_2"

images_paths = sorted(glob.glob(f"{images_path}/*"))
labels_paths = sorted(glob.glob(f"{labels_path}/*"))

length_images = len(images_paths)
length_labels = len(labels_paths)

half_length_images = length_images // 2
half_length_labels = length_labels // 2

for i, img_path in enumerate(images_paths):
    if i < half_length_images:
        dest_dir = half_1_directory + "/images"
    else:
        dest_dir = half_2_directory + "/images"
    
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(img_path, dest_dir)

for i, label_path in enumerate(labels_paths):
    if i < half_length_labels:
        dest_dir = half_1_directory + "/labels"
    else:
        dest_dir = half_2_directory + "/labels"
    
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(label_path, dest_dir)

