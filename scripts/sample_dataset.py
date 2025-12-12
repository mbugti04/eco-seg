"""
Docstring for scripts.sample_dataset

Samples a specified number of images from each site and deployment.
"""

import os
import glob
import shutil
import numpy as np

dataset_name = "increase_dataset_L2"

# 1. Sampling logic (from dataset_initalizer.py)
level_to_sample = 2
directory = f"/media/research/data/flow_1024_512/label_{level_to_sample}/*"
files = glob.glob(directory)
file_dict = {}  #stores the files in a dictionary with keys as tuples of (site, deployment, label)
instances = 10   #set how many examples you want to sample from each site & deployment \

images = "images_to_process.txt"
images_path = f"/home/research/Documents/eco-seg/{dataset_name}"
os.makedirs(images_path, exist_ok=True)

images = os.path.join(images_path, images)


for f in files: 
    name_full =f.split("/")[-1]
    site = name_full.split("_")[0]         
    deployment = name_full.split("_")[1] +  "_" + name_full.split("_")[2]  
    label = name_full.split("_")[6].split(".")[0]       

    k = (site, deployment, label)
    if k not in file_dict:
        file_dict[k] = [f]
    else:
        file_dict[k] += [f]

for k in file_dict:
    n_available = len(file_dict[k])
    n_sample = min(instances, n_available) #if there are less than instances available, sample all
    x = np.random.choice(file_dict[k], n_sample, replace=False)
    
    # Read existing content first
    if os.path.exists(images):
        with open(images, "r") as f:
            content = f.read()
    else:
        content = ""
    
    # Append only new paths
    with open(images, "a") as f:
        for path in x:
            if path not in content:
                f.write(path + "\n")

    with open(images, 'r') as file:
        content = file.read()
        print(content)


# 2. Copy sampled images to UI dataset folder (from copy_images.py)
images_list_path = f'{dataset_name}/images_to_process.txt'
destination_dir = f'{dataset_name}/images'

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
