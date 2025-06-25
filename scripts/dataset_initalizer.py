import glob
import numpy as np

directory = "/media/research/data/flow_1024_512/label_6/*"
files = glob.glob(directory)
file_dict = {}              #stores the files in a dictionary with keys as tuples of (site, deployment, label)
instances = 1               #set how many examples you want to sample from each site & deployment 


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

# x = ('S15796', 'D110420_061621', 'L6')
# print(file_dict[x])

for k in file_dict:
    x = np.random.choice(file_dict[k], instances)
    print(x)

