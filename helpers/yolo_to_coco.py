#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""YOLO to COCO Annotation Converter and Visualization Tool.

This script converts YOLO-format annotations into COCO-format annotations,
optionally draws bounding boxes on the corresponding images, and can create
a CSV file summarizing the annotations.

Directory Structure:
-------------------
The input directory (data_dir) should contain:
    data_dir/
        images/
            image_1.png
            image_2.png
            ...
        labels/
            image_1.txt
            image_2.txt
            ...

Input Format:
------------
- Each .txt file in labels/ corresponds to an image in images/ with the same base name
- YOLO annotation format (one line per box):
    class_id x_center y_center width height [score]
    where coordinates are normalized (0-1) relative to image dimensions
    Note: score is optional

Output Files:
------------
1. COCO Format (--convert_to_coco):
    Creates yolo_to_coco_conversion_XXXX.json in data_dir/

2. Annotated Images (--draw_annotations):
    Creates data_dir/annotated_images/ containing:
        image_1.png
        image_2.png
        ...

3. CSV Summary (--create_csv):
    Creates data_dir/ground_truth.csv with format:
        ID,TARGET
        image_name,1.0 x1 y1 w h
        image_name,1.0 x1 y1 w h
        ...
    Note: Images without boxes listed as:
        image_name,

Usage:
------
Convert to COCO:
    python yolo_to_coco.py --data_dir <path_to_data_dir> --convert_to_coco

Draw annotations:
    python yolo_to_coco.py --data_dir <path_to_data_dir> --draw_annotations

Create CSV:
    python yolo_to_coco.py --data_dir <path_to_data_dir> --create_csv

Author: Joseph Adeola
Email: adeolajosepholoruntoba@gmail.com
Date: 2024-12-11
"""

import os
import re
import hashlib
import json
import argparse
from PIL import Image, ImageDraw
from termcolor import colored
import random
import sys


def validate_directories(data_dir):
    """
    Validate that the required directories (images and labels) exist.

    Parameters:
    - data_dir (str): Path to the base directory containing 'images' and 'labels' subdirectories.

    Raises:
    - FileNotFoundError: If the required directories or files are not found.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The provided data directory '{data_dir}' does not exist.")

    images_folderpath = os.path.join(data_dir, "images")
    labels_folderpath = os.path.join(data_dir, "labels")

    if not os.path.isdir(images_folderpath):
        raise FileNotFoundError(f"Images directory not found at '{images_folderpath}'.")
    if not os.path.isdir(labels_folderpath):
        raise FileNotFoundError(f"Labels directory not found at '{labels_folderpath}'.")


def convert_to_coco_format(data_dir, image_extensions=['jpg', 'png']):
    """
    Converts YOLO format annotations to COCO format.

    This function reads YOLO format annotation files, converts normalized bounding box coordinates 
    to absolute coordinates using the image dimensions, and saves the annotations in COCO format 
    to a JSON file in the data directory.

    Parameters:
    - data_dir (str): Path to the directory containing 'labels' and 'images' subdirectories.
    - image_extensions (list): List of possible image file extensions, default ['jpg', 'png'].

    Returns:
    - None: The function saves a COCO format JSON file to the data directory.

    COCO Format Schema:
    {
      "images": [
        {
          "id": <int>,
          "width": <int>,
          "height": <int>
        },
        ...
      ],
      "annotations": [
        {
          "id": <int>,
          "image_id": <int>,
          "category_id": <float>,
          "bbox": [x_min, y_min, width, height],
          "area": <float>,
          "iscrowd": 0,
          "score": <float>
        },
        ...
      ],
      "categories": [
        {
          "id": 1,
          "name": "lesion"
        }
      ]
    }
    """
    validate_directories(data_dir)

    coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "lesion"}]}
    annotation_id = 1

    labels_folderpath = os.path.join(data_dir, "labels")
    images_folderpath = os.path.join(data_dir, "images")

    label_files = os.listdir(labels_folderpath)
    # Sort label files based on numeric values found in filenames
    label_files.sort(key=lambda f: [int(x) for x in re.findall(r'\d+', str(f)) if x.isdigit()])

    for label_file in label_files:
        if not label_file.endswith('.txt'):
            continue

        with open(os.path.join(labels_folderpath, label_file), 'r') as f:
            lines = f.readlines()

        # Extract image_id from filename
        # Expected format: something_number.txt
        image_id_candidates = re.findall(r'\d+', label_file)
        if not image_id_candidates:
            # If no digit found, create a pseudo ID from hash
            image_id = int(hashlib.sha1(label_file.encode()).hexdigest(), 16) % (10 ** 8)
        else:
            image_id = int(image_id_candidates[-1])

        # Find corresponding image
        image_path = None
        for ext in image_extensions:
            temp_image_path = os.path.join(images_folderpath, label_file.replace('.txt', f'.{ext}'))
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break

        if image_path is None:
            # If no matching image found, skip
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                # Invalid format line, skip
                continue
            try:
                # Attempt to parse score if present
                if len(parts) == 6:
                    class_id, x, y, w, h, score = map(float, parts)
                else:
                    class_id, x, y, w, h = map(float, parts[:5])
                    score = 1.0
            except ValueError:
                # If parsing fails, skip this line
                continue

            # Convert normalized coordinates (YOLO) to absolute (COCO)
            x_abs = x * width
            y_abs = y * height
            w_abs = w * width
            h_abs = h * height
            x1 = int(x_abs - w_abs / 2)
            y1 = int(y_abs - h_abs / 2)

            area = w_abs * h_abs

            boxes.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,  # As is, assuming class_id is already the correct ID
                "bbox": [x1, y1, w_abs, h_abs],
                "area": area,
                "iscrowd": 0,
                "score": score
            })
            annotation_id += 1

        coco_data["images"].append({"id": image_id, "width": width, "height": height})
        coco_data["annotations"].extend(boxes)

    output_json_file_name = f"yolo_to_coco_conversion_{random.randint(1000,9999)}.json"
    output_json_file = os.path.join(data_dir, output_json_file_name)

    with open(output_json_file, 'w') as json_file:
        json.dump(coco_data, json_file)

    print(colored(f"COCO format JSON file saved at: {output_json_file}", "green"))


def create_csv(data_dir, image_extensions=['jpg', 'png']):
    """
    Creates a CSV file with bounding box information from YOLO annotations.

    The CSV format:
    ID, TARGET
    image_id,1.0 x1 y1 w h
    If multiple boxes per image, each one is on a new line with the same ID.
    If an image has no bounding boxes, it will have a line:
    image_id,

    Parameters:
    - data_dir (str): Path to the directory containing 'labels' and 'images'.
    - image_extensions (list): List of possible image file extensions.

    Returns:
    - None: The function saves 'ground_truth.csv' in the data directory.
    """
    validate_directories(data_dir)

    csv_data = []

    labels_folderpath = os.path.join(data_dir, "labels")
    images_folderpath = os.path.join(data_dir, "images")
    label_files = os.listdir(labels_folderpath)
    label_files.sort(key=lambda f: [int(x) for x in re.findall(r'\d+', str(f)) if x.isdigit()])

    for label_file in label_files:
        if not label_file.endswith('.txt'):
            continue

        image_id = label_file.replace('.txt', '')

        # Find corresponding image
        image_path = None
        for ext in image_extensions:
            temp_image_path = os.path.join(images_folderpath, f"{image_id}.{ext}")
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break

        if image_path is None:
            # No image found for this label file
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        with open(os.path.join(labels_folderpath, label_file), 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            # No bounding boxes
            csv_data.append(f"{image_id},")
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                # Invalid annotation line, skip
                continue

            try:
                class_id, x, y, w, h = map(float, parts[:5])
            except ValueError:
                continue

            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            w_abs = int(w * width)
            h_abs = int(h * height)

            csv_data.append(f"{image_id},1.0 {x1} {y1} {w_abs} {h_abs}")

    # Write the CSV file
    csv_file_path = os.path.join(data_dir, "ground_truth.csv")
    with open(csv_file_path, 'w') as csv_file:
        csv_file.write("ID,TARGET\n")
        for entry in csv_data:
            csv_file.write(entry + "\n")

    print(colored(f"CSV file saved at: {csv_file_path}", "green"))


def draw_annotations(data_dir, image_extensions=['jpg', 'png']):
    """
    Draws bounding boxes on images based on YOLO format annotations and saves them.

    The annotated images are stored in 'annotated_images' directory within data_dir.

    Parameters:
    - data_dir (str): Path to the directory containing 'labels' and 'images' subdirectories.
    - image_extensions (list): List of possible image file extensions.

    Returns:
    - None: Saves annotated images in the `annotated_images` directory.
    """
    validate_directories(data_dir)

    labels_folderpath = os.path.join(data_dir, "labels")
    images_folderpath = os.path.join(data_dir, "images")

    label_files = os.listdir(labels_folderpath)
    label_files.sort(key=lambda f: [int(x) for x in re.findall(r'\d+', str(f)) if x.isdigit()])

    annotated_images_dir = os.path.join(data_dir, "annotated_images")
    if not os.path.exists(annotated_images_dir):
        os.makedirs(annotated_images_dir)

    for label_file in label_files:
        if not label_file.endswith('.txt'):
            continue

        image_path = None
        for ext in image_extensions:
            temp_image_path = os.path.join(images_folderpath, label_file.replace('.txt', f'.{ext}'))
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break

        if image_path is None:
            # No corresponding image found, skip
            continue

        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)

            with open(os.path.join(labels_folderpath, label_file), 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    # Invalid annotation line
                    continue
                try:
                    class_id, x, y, w, h = map(float, parts[:5])
                except ValueError:
                    continue

                x_abs = x * img.size[0]
                y_abs = y * img.size[1]
                w_abs = w * img.size[0]
                h_abs = h * img.size[1]

                x1 = int(x_abs - w_abs / 2)
                y1 = int(y_abs - h_abs / 2)
                x2 = int(x_abs + w_abs / 2)
                y2 = int(y_abs + h_abs / 2)

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)

            image_name = label_file.replace('.txt', '')
            for ext in image_extensions:
                if image_path.endswith(ext):
                    annotated_image_path = os.path.join(annotated_images_dir, f"{image_name}.{ext}")
                    img.save(annotated_image_path)
                    break

    print(colored(f"Annotated images saved in: {annotated_images_dir}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YOLO annotations: Convert to COCO, draw bounding boxes, or create CSV.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the parent directory containing 'labels' and 'images' folders")
    parser.add_argument("--convert_to_coco", action="store_true", help="Convert YOLO format to COCO format JSON file")
    parser.add_argument("--draw_annotations", action="store_true", help="Draw bounding boxes on images and save them")
    parser.add_argument("--create_csv", action="store_true", help="Create a CSV file with bounding box information")

    args = parser.parse_args()

    # Execute chosen operations
    if args.convert_to_coco:
        convert_to_coco_format(args.data_dir)
    if args.draw_annotations:
        draw_annotations(args.data_dir)
    if args.create_csv:
        create_csv(args.data_dir)

    # If no action specified, print help
    if not (args.convert_to_coco or args.draw_annotations or args.create_csv):
        print("No operation selected. Use --convert_to_coco, --draw_annotations, or --create_csv.")
        sys.exit(1)
