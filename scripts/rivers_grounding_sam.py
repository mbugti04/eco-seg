import os

os.environ["KERAS_BACKEND"] = "tensorflow"


import numpy as np
import matplotlib.pyplot as plt
import keras
# keras.mixed_precision.set_global_policy("mixed_float16")
print(keras.mixed_precision.dtype_policy)
from keras import ops
import keras_hub
from groundingdino.util.inference import Model as GroundingDINO
from PIL import Image
import glob
import json

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
