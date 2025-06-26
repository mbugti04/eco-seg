import os
import glob
import shutil
import numpy as np
import json
from keras import ops

# 1. Sampling logic (from dataset_initalizer.py)
directory = "/media/research/data/flow_1024_512/label_6/*"
files = glob.glob(directory)
file_dict = {}              #stores the files in a dictionary with keys as tuples of (site, deployment, label)
instances = 1              #set how many examples you want to sample from each site & deployment \

try:
    os.remove("/home/research/Documents/eco-seg/example_dataset/images_to_process.txt")  #remove the file if it already exists
except OSError:
    pass

images = "images_to_process.txt"
images_path = "/home/research/Documents/eco-seg/example_dataset"
images = os.path.join(images_path, images)

directory = "/media/research/data/flow_1024_512/label_6/*"
files = glob.glob(directory)
file_dict = {}
instances = 1

for f in files:
    name_full = f.split("/")[-1]
    site = name_full.split("_")[0]
    deployment = name_full.split("_")[1] + "_" + name_full.split("_")[2]
    label = name_full.split("_")[6].split(".")[0]
    k = (site, deployment, label)
    if k not in file_dict:
        file_dict[k] = [f]
    else:
        file_dict[k] += [f]

sampled_files = []
for k in file_dict:
    n_available = len(file_dict[k])
    n_sample = min(instances, n_available)
    x = np.random.choice(file_dict[k], n_sample, replace=False)
    sampled_files.extend(x)



# 2. Copy sampled images to UI dataset folder (from copy_images.py)
images_dir = "example_dataset/images"
os.makedirs(images_dir, exist_ok=True)
for img_path in sampled_files:
    if os.path.isfile(img_path):
        shutil.copy(img_path, images_dir)
    else:
        print(f"Warning: File not found - {img_path}")


# 3. Run DINO+SAM and save masks (from rivers_grounding_sam.py)
from PIL import Image
import keras
import keras_hub
from groundingdino.util.inference import Model as GroundingDINO
from rivers_grounding_sam_helper import inference_resizing, unpad_and_resize

from rivers_grounding_sam_helper import show_mask, show_points, show_box, inference_resizing, unpad_and_resize

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
image_paths = glob.glob("example_dataset/images/*.*")
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


    # Specify what to segment
    # foliage (trees, shrubs, grasses), rocks and sediment, river beds (the areas carved by water that transition from earth to water), sky, sun
    object_to_segment = "river"
    boxes = grounding_dino_model.predict_with_caption(image.astype(np.uint8), object_to_segment)
    boxes = np.array(boxes[0].xyxy)

    if boxes.size == 0:
        raise Exception(f"Grounding DINO did not find any bounding boxes for object '{object_to_segment}'") 

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
