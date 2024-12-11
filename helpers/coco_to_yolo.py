#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""COCO to YOLO Annotation Converter.

This script converts COCO-format annotations into YOLO-format annotations.

Directory Structure:
-------------------
Input directory (data_dir) should contain:
   data_dir/
       images/
           image_1.png
           image_2.png
           ...
       coco_annotations.json

Input Format:
------------
coco_annotations.json structure:
   {
       "images": [
           {
               "id": <int>,
               "file_name": <str>,
               "width": <int>,
               "height": <int>
           },
           ...
       ],
       "annotations": [
           {
               "image_id": <int>,
               "category_id": <int>,
               "bbox": [x_min, y_min, width, height],
               "iscrowd": 0 or 1,
               ...
           },
           ...
       ],
       "categories": [
           {
               "id": <int>,
               "name": <str>
           },
           ...
       ]
   }

Output Format:
-------------
Creates data_dir/labels/ containing:
   image_1.txt
   image_2.txt
   ...

Each .txt file contains YOLO annotations:
   class_id x_center y_center width height
   where coordinates are normalized (0-1) relative to image dimensions

Usage:
------
python coco_to_yolo.py --data_dir <path_to_data_dir> --coco_json <path_to_coco_json>

Notes:
------
Category IDs are preserved from COCO format. For standard YOLO format (starting at 0),
remap category IDs as needed. Current implementation assumes category IDs start at 1
and are contiguous.

Author: Joseph Adeola
Email: adeolajosepholoruntoba@gmail.com
Date: 2024-12-11
"""

import os
import json
import argparse
from termcolor import colored
import sys


def validate_input(data_dir, coco_json):
    """
    Validate that the required directories and COCO JSON file exist.

    Parameters:
    - data_dir (str): Path to the base directory containing images.
    - coco_json (str): Path to the COCO annotation JSON file.

    Raises:
    - FileNotFoundError: If the data directory or the coco_json file does not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The provided data directory '{data_dir}' does not exist.")
    if not os.path.isfile(coco_json):
        raise FileNotFoundError(f"The COCO JSON file '{coco_json}' does not exist.")

    images_folder = os.path.join(data_dir, "images")
    if not os.path.isdir(images_folder):
        raise FileNotFoundError(f"Images directory not found at '{images_folder}'.")


def load_coco_annotations(coco_json):
    """
    Load and parse the COCO annotations from a JSON file.

    Parameters:
    - coco_json (str): Path to the COCO-format JSON file.

    Returns:
    - coco_data (dict): Parsed COCO data containing images, annotations, categories.
    """
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # Basic validation of the COCO structure
    if "images" not in coco_data or "annotations" not in coco_data or "categories" not in coco_data:
        raise ValueError("COCO JSON missing required keys (images, annotations, categories).")

    return coco_data


def create_yolo_annotations(data_dir, coco_data):
    """
    Convert COCO annotations to YOLO format and save them as .txt files in the labels directory.

    YOLO format for each bounding box line:
    class_id x_center y_center width height
    - class_id: zero-based index of the category
    - x_center, y_center, width, height: normalized [0,1] by image width and height

    Parameters:
    - data_dir (str): Path to the directory containing images.
    - coco_data (dict): Loaded COCO data dictionary.

    Returns:
    - None: Writes YOLO annotation .txt files into data_dir/labels
    """
    # Create labels directory if not exists
    labels_dir = os.path.join(data_dir, "labels")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Build a map from image_id to image info (width, height, file_name)
    image_info_map = {img["id"]: img for img in coco_data["images"]}

    # Build a category_id to yolo_class_id map
    # Assuming category IDs in COCO start from 1 and are contiguous
    category_ids = [cat["id"] for cat in coco_data["categories"]]
    category_ids.sort()
    cat_to_yolo = {cat_id: i for i, cat_id in enumerate(category_ids)}

    # Prepare a structure to hold annotations per image
    annotations_per_image = {img["id"]: [] for img in coco_data["images"]}

    # Populate annotations per image
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_per_image:
            # It's possible an annotation refers to an image not listed; skip if so
            continue

        if ann.get("iscrowd", 0) == 1:
            # YOLO typically doesn't handle "crowd" annotations directly, skip if needed
            # or handle them differently if your use case requires
            continue

        cat_id = ann["category_id"]
        if cat_id not in cat_to_yolo:
            # If category is not known or not mapped, skip
            continue
        
        yolo_class_id = cat_to_yolo[cat_id]

        # COCO bbox: [x_min, y_min, width, height]
        x_min, y_min, w, h = ann["bbox"]
        
        # Get image dimensions
        img_width = image_info_map[image_id]["width"]
        img_height = image_info_map[image_id]["height"]

        # Convert COCO bbox to YOLO bbox
        # YOLO expects normalized center_x, center_y, width, height
        x_center = x_min + w/2.0
        y_center = y_min + h/2.0

        # Normalize by image dimensions
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        annotations_per_image[image_id].append(
            f"{yolo_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    # Write out YOLO annotation files
    for img in coco_data["images"]:
        image_id = img["id"]
        file_name = img["file_name"]
        base_name, _ = os.path.splitext(file_name)

        yolo_file_path = os.path.join(labels_dir, f"{base_name}.txt")
        with open(yolo_file_path, 'w') as yf:
            # If no annotations, file will be blank
            if image_id in annotations_per_image and annotations_per_image[image_id]:
                for line in annotations_per_image[image_id]:
                    yf.write(line + "\n")

    print(colored(f"YOLO annotations saved in: {labels_dir}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO format.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the base directory containing images and coco_annotations.json")
    parser.add_argument("--coco_json", type=str, required=True, help="Path to the COCO annotation JSON file")

    args = parser.parse_args()

    try:
        validate_input(args.data_dir, args.coco_json)
        coco_data = load_coco_annotations(args.coco_json)
        create_yolo_annotations(args.data_dir, coco_data)
    except Exception as e:
        print(colored(f"Error: {e}", "red"))
        sys.exit(1)
