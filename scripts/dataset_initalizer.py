import glob
import numpy as np
import os

directory = "/media/research/data/flow_1024_512/label_6/*"
files = glob.glob(directory)
file_dict = {}              #stores the files in a dictionary with keys as tuples of (site, deployment, label)
instances = 1              #set how many examples you want to sample from each site & deployment \

try:
    os.remove("/home/research/Documents/eco-seg/example_dataset/images_to_process.txt")  #remove the file if it already exists
except OSError:
    print("File does not exist, creating a new one.")

images = "images_to_process.txt"
images_path = "/home/research/Documents/eco-seg/example_dataset"
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
    n_sample = min(instances, n_available)      #if there are less than instances available, sample all
    x = np.random.choice(file_dict[k], n_sample, replace=False)
    with open(images, "a") as f:                #save the path of the sampled images to a text file
        for path in x:
            f.write(path + "\n")



