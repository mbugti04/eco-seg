"""Generate precomputed SAM region masks for an EcoSeg dataset.

The GUI reads masks from ``<dataset>/sam/<image-stem>.png`` when Model
assistance is enabled. This script supports the same common image formats as
the GUI, so an image does not need to be renamed to ``.png`` first.
"""

import argparse
from pathlib import Path
import time
import tomllib

import numpy as np


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def make_annotator(weights_path: str, device: str):
    """Load SAM only when mask generation is actually requested."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    model_type = "vit_h"
    print(f"Loading {model_type} on {device} device")
    t1 = time.perf_counter()
    sam = sam_model_registry[model_type](weights_path)
    t2 = time.perf_counter()
    sam.to(device)
    t3 = time.perf_counter()
    mask_generator = SamAutomaticMaskGenerator(sam)
    print(f"Load weights: {(t2 - t1):.3f}s\nMove to {device}: {(t3 - t2):.3f}s")
    return mask_generator


def image_paths(images_path: Path) -> list[Path]:
    """Return supported image files in a stable order."""
    return sorted(
        path
        for path in images_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate precomputed SAM region masks for an EcoSeg dataset."
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--data", type=Path, help="Dataset containing images/")
    parser.add_argument("--weights", type=Path, help="SAM checkpoint path")
    parser.add_argument("--device", help="Torch device, such as cuda or cpu")
    return parser


def resolve_options(args: argparse.Namespace) -> tuple[Path, Path, str]:
    config_path = args.config.expanduser().resolve()
    config = {}
    if config_path.exists():
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)

    data_value = args.data or config.get("paths", {}).get("data")
    weights_value = args.weights or config.get("paths", {}).get("sam_weights")
    device = args.device or config.get("device", "cuda")
    if data_value is None or weights_value is None:
        raise ValueError("Provide --data and --weights, or set both in config.toml.")

    data_path = Path(data_value).expanduser()
    weights_path = Path(weights_value).expanduser()
    if not data_path.is_absolute():
        data_path = config_path.parent / data_path
    if not weights_path.is_absolute():
        weights_path = config_path.parent / weights_path
    return data_path.resolve(), weights_path.resolve(), device


def main(argv: list[str] | None = None) -> int:
    from PIL import Image
    from tqdm import tqdm

    args = build_parser().parse_args(argv)
    try:
        data_path, weights_path, device = resolve_options(args)
    except ValueError as exc:
        build_parser().error(str(exc))

    images_path = data_path / "images"
    if not images_path.is_dir():
        build_parser().error(f"Dataset is missing an images directory: {images_path}")
    if not weights_path.is_file():
        build_parser().error(f"SAM weights not found: {weights_path}")

    sam_path = data_path / "sam"
    sam_path.mkdir(exist_ok=True)
    sam = make_annotator(str(weights_path), device)

    for img_path in tqdm(image_paths(images_path)):
        out_path = sam_path / f"{img_path.stem}.png"
        img = Image.open(img_path)
        orig_size = img.size
        low_res_size = (max(1, orig_size[0] // 4), max(1, orig_size[1] // 4))
        img_low_res = img.resize(low_res_size, resample=Image.Resampling.BILINEAR)
        masks = sam.generate(np.array(img_low_res))
        if not masks:
            print(f"No masks generated for {img_path.name}; skipping.")
            continue

        sorted_masks = sorted(masks, key=lambda mask: mask["area"], reverse=True)
        label = np.zeros(sorted_masks[0]["segmentation"].shape, dtype=np.uint8)
        for index, mask in enumerate(sorted_masks[:255], start=1):
            label[mask["segmentation"]] = index

        label_img = Image.fromarray(label, mode="L")
        label_img.resize(orig_size, resample=Image.Resampling.NEAREST).save(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
