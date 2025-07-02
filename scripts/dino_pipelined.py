import os
import glob
import shutil
import numpy as np
import json
from keras import ops
from PIL import Image
import keras
import keras_hub
from groundingdino.util.inference import Model as GroundingDINO
from rivers_grounding_sam_helper import inference_resizing, unpad_and_resize, inference_resizing, unpad_and_resize


# 1. Sampling logic (from dataset_initalizer.py)
directory = "/media/research/data/flow_1024_512/label_5/*"
files = glob.glob(directory)
file_dict = {}              #stores the files in a dictionary with keys as tuples of (site, deployment, label)
instances = 1               #set how many examples you want to sample from each site & deployment \

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


# 3. Run DINO+SAM and save masks (from rivers_grounding_sam.py)
def sam_segmentor():
    sam_model = keras_hub.models.SAMImageSegmenter.from_preset("sam_huge_sa1b")
    return sam_model

def grounding_dino_annotator():
    CONFIG_PATH = "model_weights/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "model_weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = GroundingDINO(CONFIG_PATH, WEIGHTS_PATH)
    return grounding_dino_model

# Import models
grounding_dino_model = grounding_dino_annotator()
sam_model = sam_segmentor()

# Get images from path
with open(images, "r") as f:
    image_paths = [line.strip() for line in f if line.strip()]

print(image_paths)

# Create directory for masks
masks_dir = "example_dataset/sam"
os.makedirs(masks_dir, exist_ok=True)

# Load class names from classes.json
with open("example_dataset/classes.json", "r") as f:
    classes_data = json.load(f)
object_list = [cls["name"] for cls in classes_data["classes"]]

# Segment each image
for image_index in range(len(image_paths)):
    # Preprocess images
    image = np.array(keras.utils.load_img(image_paths[image_index]))
    print(image_paths[image_index])

    original_shape = image.shape

    resized_image, preprocess_shape = inference_resizing(image)
    image_np = ops.convert_to_numpy(resized_image)
    image = image_np


    # Obtain Grounding DINO weights
    # !!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    # !!wget -q https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/v0.1.0-alpha2/groundingdino/config/GroundingDINO_SwinT_OGC.py


    object_to_segment = "river"
    boxes = grounding_dino_model.predict_with_caption(image.astype(np.uint8), object_to_segment)
    boxes = np.array(boxes[0].xyxy)

    if boxes.size == 0:
        print(f"Grounding DINO did not find any bounding boxes for object '{object_to_segment}'") 
        continue 

    outputs = sam_model.predict(
        {
            "images": np.repeat(image[np.newaxis, ...], boxes.shape[0], axis=0),
            "boxes": boxes.reshape(-1, 1, 2, 2),
        },
        batch_size=1,
    )

    mask = outputs["masks"][0]
    mask, *_ = inference_resizing(mask[0][..., None], pad=False)
    mask = mask[..., 0]
    mask = ops.convert_to_numpy(mask) > 0.0

    print("shape:", mask.shape, preprocess_shape, original_shape)

    # Unpad and convert to image
    mask = unpad_and_resize(mask, preprocess_shape, original_shape)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    
    # Save the mask image in the masks directory
    imgpath = image_paths[image_index]
    file_name = imgpath[imgpath.rfind('/')+1:imgpath.rfind('.')] + ".png"
    mask_path = os.path.join(masks_dir, file_name)
    mask_img.save(mask_path)
