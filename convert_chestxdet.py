"""
Convert ChestX-Det dataset to the per-image JSON format
expected by train_sam3_lora.py's SAM3Dataset.

ChestX-Det format (single JSON array):
  [{"file_name": "36346.png", "syms": ["Cardiomegaly", "Effusion"],
    "boxes": [[x1,y1,x2,y2], ...], "polygons": [[[x,y],...], ...]}, ...]

SAM3Dataset expected format:
  data/train/images/36346_cardiomegaly.png
  data/train/annotations/36346_cardiomegaly.json
    {"text_prompt": "cardiomegaly", "masks": [[...]], "bboxes": [[x1,y1,x2,y2]]}

Each image-category pair becomes a separate training example because
SAM3Dataset expects one text_prompt per annotation file.

Usage:
    python convert_chestxdet.py \
        --annotations ChestX_Det_train.json \
        --images path/to/train_images/ \
        --output data/train

    python convert_chestxdet.py \
        --annotations ChestX_Det_test.json \
        --images path/to/test_images/ \
        --output data/val
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Map ChestX-Det category names to SAM3 text prompts.
# These must match the prompts in gemma/keyword_mapping.py.
CATEGORY_TO_PROMPT = {
    "Atelectasis": "atelectasis",
    "Calcification": "calcification",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Diffuse Nodule": "lung nodule",
    "Effusion": "pleural effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "pulmonary fibrosis",
    "Fracture": "rib fracture",
    "Mass": "lung mass",
    "Nodule": "lung nodule",
    "Pleural Thickening": "pleural thickening",
    "Pneumothorax": "pneumothorax",
}


def polygon_to_mask(polygon: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Convert a polygon contour to a binary mask (list of lists of 0/1)."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    flat_points = [(p[0], p[1]) for p in polygon]
    if len(flat_points) >= 3:
        draw.polygon(flat_points, fill=1)
    mask = np.array(img).tolist()
    return mask


def convert(annotations_path: str, images_dir: str, output_dir: str) -> None:
    with open(annotations_path, "r") as f:
        entries = json.load(f)

    out_images = Path(output_dir) / "images"
    out_annotations = Path(output_dir) / "annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    out_annotations.mkdir(parents=True, exist_ok=True)

    images_dir = Path(images_dir)
    total = 0
    skipped = 0

    for entry in entries:
        file_name = entry["file_name"]
        syms = entry.get("syms", [])
        boxes = entry.get("boxes", [])
        polygons = entry.get("polygons", [])

        if not syms:
            continue

        src_image = images_dir / file_name
        if not src_image.exists():
            skipped += 1
            continue

        # Get image dimensions for polygon-to-mask conversion.
        with Image.open(src_image) as img:
            width, height = img.size

        # Group annotations by category so each training example
        # has one text_prompt with all its masks/boxes.
        category_groups: dict[str, dict] = {}
        for sym, box, polygon in zip(syms, boxes, polygons):
            prompt = CATEGORY_TO_PROMPT.get(sym)
            if prompt is None:
                continue
            if prompt not in category_groups:
                category_groups[prompt] = {"bboxes": [], "masks": []}
            category_groups[prompt]["bboxes"].append(box)
            category_groups[prompt]["masks"].append(
                polygon_to_mask(polygon, width, height)
            )

        stem = Path(file_name).stem

        for prompt, data in category_groups.items():
            # Sanitize prompt for filename.
            safe_prompt = prompt.replace(" ", "_")
            example_name = f"{stem}_{safe_prompt}"

            # Copy image with unique name.
            dst_image = out_images / f"{example_name}.png"
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)

            # Write annotation JSON.
            annotation = {
                "text_prompt": prompt,
                "bboxes": data["bboxes"],
                "masks": data["masks"],
            }
            ann_path = out_annotations / f"{example_name}.json"
            with open(ann_path, "w") as f:
                json.dump(annotation, f)

            total += 1

    print(f"Created {total} training examples in {output_dir}")
    if skipped:
        print(f"Skipped {skipped} entries (image not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ChestX-Det to SAM3 training format"
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to ChestX_Det_train.json or ChestX_Det_test.json",
    )
    parser.add_argument(
        "--images", required=True,
        help="Path to directory containing the CXR images",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory (e.g. data/train or data/val)",
    )
    args = parser.parse_args()
    convert(args.annotations, args.images, args.output)
