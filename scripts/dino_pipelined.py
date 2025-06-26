import os
import glob
import shutil
import numpy as np
import json

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

# Load models
def sam_segmentor():
    return keras_hub.models.SAMImageSegmenter.from_preset("sam_huge_sa1b")
def grounding_dino_annotator():
    CONFIG_PATH = "model_weights/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = "model_weights/groundingdino_swint_ogc.pth"
    return GroundingDINO(CONFIG_PATH, WEIGHTS_PATH)

grounding_dino_model = grounding_dino_annotator()
sam_model = sam_segmentor()

# Load class names
with open("example_dataset/classes.json", "r") as f:
    classes_data = json.load(f)
object_list = [cls["name"] for cls in classes_data["classes"]]

# Create directory for masks
masks_dir = "example_dataset/sam"
os.makedirs(masks_dir, exist_ok=True)

# Segment each image
image_paths = glob.glob(os.path.join(images_dir, "*.*"))
for image_path in image_paths:
    image = np.array(keras.utils.load_img(image_path))
    original_shape = image.shape
    resized_image, preprocess_shape = inference_resizing(image)
    image_np = keras.ops.convert_to_numpy(resized_image)
    image = image_np

    combined_mask = np.zeros(original_shape[:2], dtype=np.uint8)
    for class_idx, object_to_segment in enumerate(object_list, start=1):
        boxes = grounding_dino_model.predict_with_caption(image.astype(np.uint8), object_to_segment)
        boxes = np.array(boxes[0].xyxy)
        if boxes.size == 0:
            print(f"Warning: No boxes found for '{object_to_segment}' in {image_path}")
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
        mask = keras.ops.convert_to_numpy(mask) > 0.0
        mask = unpad_and_resize(mask, preprocess_shape, original_shape)
        combined_mask[mask] = class_idx

    # Save the combined mask
    file_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    mask_path = os.path.join(masks_dir, file_name)
    Image.fromarray(combined_mask.astype(np.uint8), mode="L").save(mask_path)