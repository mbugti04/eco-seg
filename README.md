
# EcoSeg

EcoSeg is a semantic segmentation tool built off of SAMAT, tailored to quickly and efficiently segment environmental images. The project focuses on stream and river scene annotation, with support for large dataset handling and sampling.

EcoSeg samples images from a dataset through specific sites (and its deployments), balancing the dataset among the six stream connectivity labels through augmentations.

## Getting started

### Prerequisites

Core GUI requirements:

- `Python 3.11`
- `PyQt5`
- `numpy`

Requirements for segmentation pipeline:

- `keras`
- `tensorflow`
- `segment_anything`
- `GroundingDino`

Example setup (Ubuntu):

```bash
git clone https://github.com/mbugti04/eco-seg.git
cd eco-seg
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -e .
python .
```

### Dataset folder structure

For automatic sampling, define the folder to sample from in `dino_pipelined.py`. The script will automatically create a ready-to-use dataset following this structure:

```
── my_dataset
   ├── images
   |   ├── 000001.png
   |   ├── 000002.png
   |   └── ...
   ├── labels (optional)
   |   ├── 000001.png
   |   ├── 000002.png
   |   └── ...
   ├── sam (optional)
   |   ├── 000001.png
   |   ├── 000002.png
   |   └── ...
   └── classes.json
```

- `images` contains `.png` (or other image file format) files you want to label
- `labels` contains `.png` (or other image file format) files with labels (will be automatically created if you have no labels yet)
- `sam` contains `.png` files with SAM annotations (Binary output of `dino_pipelined.py`)
- `classes.json` contains classes description that will be used for labeling

Example `classes.json`:

```json
{
    "classes": [
        { "id": 1, "name": "water", "color": "#FF0000" },
        { "id": 2, "name": "foliage", "color": "#00FF00" },
        { "id": 3, "name": "rocks", "color": "#0000FF" }
    ]
}
```

where:

- `id` field must coinside with number keys on keyboard, so start with 1 (not 0). Any number of classes allowed, but only first 9 have their shortcuts.
- `name` field is arbitrary and used only for dispaly in GUI
- `color` field specifies the color this class would be displayed in GUI and encoded in output label `.png`

**Note:** specify path to your `my_dataset` (or any other name) inside `config.toml`.

Example:

```toml
device = "cuda" # or "cpu"

[paths]
data = "example_dataset"
sam_weights = "/path/to/sam_weights.pth"
```

**Note:** Any file format is supported. Labels are saved as `<image_stem>_label.png`

## Shortcuts

|                Shortcut               | Description                                          |
| :------------------------------------:| ---------------------------------------------------- |
|           Left Mouse Button           | Draw with brush + fill region (in SAM mode)          |
|           Right Mouse Button          | Pan motion on zoomed-in image                        |
|              Mouse Wheel              | Zoom in/out                                          |
|          `Ctrl` + Mouse Wheel         | Change brush size                                    |
|                `1`-`9`                | Select class (color to draw on label layer)          |
|                  `E`                  | Eraser tool (transparent brush)                      |
|                `Space`                | Reset zoom                                           |
|                  `C`                  | Clear label                                          |
|                  `S`                  | Switch SAM assistance mode on/off                    |
|               `,`/`.`                 | Previous/Next sample                                 |
