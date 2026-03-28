"""
Convert ChestX-Det dataset to COCO format for train_sam3_lora_with_categories.py.

ChestX-Det format (single JSON array):
  [{"file_name": "36346.png", "syms": ["Cardiomegaly", "Effusion"],
    "boxes": [[x1,y1,x2,y2], ...], "polygons": [[[x,y],...], ...]}, ...]

COCO format output:
  _annotations.coco.json with categories, images, and annotations
  (polygons as segmentation, boxes converted from xyxy to xywh)

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

from PIL import Image


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


def convert(annotations_path: str, images_dir: str, output_dir: str) -> None:
    with open(annotations_path, "r") as f:
        entries = json.load(f)

    out_images = Path(output_dir) / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    # Build COCO category list from the prompt mapping.
    # Use the SAM3 prompt as the category name so training uses
    # the same text prompts as inference.
    prompt_to_cat_id: dict[str, int] = {}
    coco_categories = []
    for idx, prompt in enumerate(sorted(set(CATEGORY_TO_PROMPT.values())), start=1):
        prompt_to_cat_id[prompt] = idx
        coco_categories.append({"id": idx, "name": prompt})

    coco_images = []
    coco_annotations = []
    image_id = 0
    ann_id = 0
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

        # Get image dimensions.
        with Image.open(src_image) as img:
            width, height = img.size

        # Copy image to output directory.
        dst_image = out_images / file_name
        if not dst_image.exists():
            shutil.copy2(src_image, dst_image)

        coco_images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        for sym, box, polygon in zip(syms, boxes, polygons):
            prompt = CATEGORY_TO_PROMPT.get(sym)
            if prompt is None:
                continue

            cat_id = prompt_to_cat_id[prompt]

            # Convert box from [x1, y1, x2, y2] to COCO [x, y, w, h].
            x1, y1, x2, y2 = box
            coco_box = [x1, y1, x2 - x1, y2 - y1]

            # Flatten polygon [[x,y],[x,y],...] to [x,y,x,y,...] for COCO.
            flat_seg = []
            for point in polygon:
                flat_seg.extend(point)

            area = coco_box[2] * coco_box[3]

            coco_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": coco_box,
                "segmentation": [flat_seg],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        image_id += 1

    coco_json = {
        "categories": coco_categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    out_path = Path(output_dir) / "_annotations.coco.json"
    with open(out_path, "w") as f:
        json.dump(coco_json, f)

    print(f"Created COCO file at {out_path}")
    print(f"  {len(coco_images)} images, {len(coco_annotations)} annotations, {len(coco_categories)} categories")
    if skipped:
        print(f"  Skipped {skipped} entries (image not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ChestX-Det to COCO format for SAM3 training"
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
