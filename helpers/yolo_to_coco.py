'''
Script to convert YOLO format annotations to COCO format, draw annotations on images, and create a CSV file with bounding box information.
It was designed to create the iToBoS challenge dataset

This script provides three functionalities:
1. Converts YOLO format annotations to COCO format, saving the output to a JSON file.
2. Draws bounding boxes on images based on YOLO format annotations and saves the annotated images in a new directory.
3. Creates a CSV file containing bounding box information from YOLO format annotations.

To run this script, use the following commands:
- To convert YOLO format to COCO format: python yolo_to_coco.py --data_dir <path_to_data_dir> --convert_to_coco
- To draw annotations on images: python yolo_to_coco.py --data_dir <path_to_data_dir> --draw_annotations
- To create a CSV file: python yolo_to_coco.py --data_dir <path_to_data_dir> --create_csv

Author: Joseph Adeola
Date: 2024-09-17
'''

import os
import re
import hashlib
import json
import argparse
from PIL import Image, ImageDraw
from termcolor import colored
import random

# Function to convert YOLO format to COCO format
def convert_to_coco_format(data_dir, image_extensions=['jpg', 'png']):
    """
    Converts YOLO format annotations to COCO format.

    This function reads YOLO format annotation files, converts the normalized bounding box coordinates to absolute coordinates using the image dimensions, and saves the annotations in COCO format to a JSON file.

    Parameters:
    - data_dir (str): Path to the directory containing 'labels' and 'images' subdirectories.
    - image_extensions (list): List of possible image file extensions.

    Returns:
    - None: This function does not return a value. It saves the COCO format data to a JSON file.
    """
    # Initialize COCO data dictionary
    coco_data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "lesion"}]}
    # Initialize annotation id
    annotation_id = 1
    # Initialize score
    score = 1.0

    # Define paths for labels and images
    labels_folderpath = os.path.join(data_dir, "labels")
    images_folderpath = os.path.join(data_dir, "images")
    # Get list of label files and sort them
    label_files = os.listdir(labels_folderpath)
    label_files.sort(key=lambda f: [int(x) for x in re.findall(r'\d+', str(f))])

    # Iterate over each label file
    for label_file in label_files:
        # Open the label file and read its lines
        with open(os.path.join(labels_folderpath, label_file), 'r') as f:
            lines = f.readlines()
        # Get the image number from the label file name
        image_id = int(label_file.split('_')[-1].split('.')[0])
        # Generate image id from label file name
        # image_id = int(hashlib.sha1(label_file.split('.')[0].encode()).hexdigest(), 16) % (10 ** 8)


        # Initialize list to store bounding boxes
        boxes = []
        # Find the corresponding image file with any of the specified extensions
        image_path = None
        for ext in image_extensions:
            temp_image_path = os.path.join(images_folderpath, label_file.replace('.txt', f'.{ext}'))
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break
        if not image_path:
            continue  # Skip if no image file found with any of the specified extensions
        # Open the corresponding image to get its dimensions
        with Image.open(image_path) as img:
            width, height = img.size

        # Iterate over each line in the label file
        for line in lines:
            parts = line.split()
            try:
                class_id, x, y, w, h, score = map(float, parts[:])
            except:
                class_id, x, y, w, h = map(float, parts[:5])
            # Convert normalized coordinates to absolute coordinates using the image dimensions
            x, y, w, h = x * width, y * height, w * width, h * height
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            # Calculate area of the bounding box
            area = w * h
            # Add the bounding box to the list
            boxes.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,# + 1,
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0,
                "score": score
            })
            # Increment annotation id
            annotation_id += 1

        # Add image data to the COCO data
        coco_data["images"].append({"id": image_id, "width": width, "height": height})
        # Add bounding boxes to the COCO data
        coco_data["annotations"].extend(boxes)

    # Generate a meaningful output file name
    output_json_file_name = f"yolo_to_coco_conversion_{random.randint(1000,9999)}.json"
    output_json_file = os.path.join(data_dir, output_json_file_name)
    # Save the COCO data to a JSON file
    with open(output_json_file, 'w') as json_file:
        json.dump(coco_data, json_file)

    # Print the path of the saved JSON file
    print(colored(f"Output JSON file saved at: {output_json_file}", "green"))


def create_csv(data_dir, image_extensions=['jpg', 'png']):
    """
    Creates a CSV file with bounding box information.

    Parameters:
    - data_dir (str): Path to the directory where the CSV will be saved.
    - image_extensions (list): List of possible image file extensions.

    Returns:
    - None: This function saves the CSV file to the specified directory.
    """
    csv_data = []

    labels_folderpath = os.path.join(data_dir, "labels")
    images_folderpath = os.path.join(data_dir, "images")
    label_files = os.listdir(labels_folderpath)
    label_files.sort(key=lambda f: [int(x) for x in re.findall(r'\d+', str(f))])

    for label_file in label_files:
        image_id = label_file.replace('.txt', '')

        # Load the corresponding image to get its dimensions
        image_path = None
        for ext in image_extensions:
            temp_image_path = os.path.join(images_folderpath, label_file.replace('.txt', f'.{ext}'))
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break
        if not image_path:
            continue  # Skip if no image file found with any of the specified extensions
        with Image.open(image_path) as img:
            width, height = img.size

        with open(os.path.join(labels_folderpath, label_file), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split()
            class_id, x, y, w, h = map(float, parts[:5])
            # Unnormalize the coordinates
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            w = int(w * width)
            h = int(h * height)
            # Append each bounding box as a new row
            csv_data.append(f"{image_id},1.0 {x1} {y1} {w} {h}")

        # If no bounding boxes, add an empty entry for the image
        if not lines:
            csv_data.append(f"{image_id},")

    # Write the CSV file
    csv_file_path = os.path.join(data_dir, "ground_truth.csv")
    with open(csv_file_path, 'w') as csv_file:
        # Write header
        csv_file.write("ID, TARGET\n")
        # Write data
        for entry in csv_data:
            csv_file.write(entry + "\n")

    print(colored(f"CSV file saved at: {csv_file_path}", "green"))
    
# Function to draw annotations on images
def draw_annotations(data_dir, image_extensions=['jpg', 'png']):
    """
    Draws bounding boxes on images based on YOLO format annotations.

    This function reads YOLO format annotation files, converts the normalized bounding box coordinates to absolute coordinates using the image dimensions, and draws the bounding boxes on the corresponding images.

    Parameters:
    - data_dir (str): Path to the directory containing 'labels' and 'images' subdirectories.
    - image_extensions (list): List of possible image file extensions.

    Returns:
    - None: This function does not return a value. It saves the annotated images to a new directory.
    """
    # Define paths for labels and images
    labels_folderpath = os.path.join(data_dir, "labels")
    images_folderpath = os.path.join(data_dir, "images")
    # Get list of label files and sort them
    label_files = os.listdir(labels_folderpath)
    label_files.sort(key=lambda f: [int(x) for x in re.findall(r'\d+', str(f))])
    # Define directory to save annotated images
    annotated_images_dir = os.path.join(data_dir, "annotated_images")
    # Create the directory if it doesn't exist
    if not os.path.exists(annotated_images_dir):
        os.makedirs(annotated_images_dir)
    # Iterate over each label file
    for label_file in label_files:
        # Find the corresponding image file with any of the specified extensions
        image_path = None
        for ext in image_extensions:
            temp_image_path = os.path.join(images_folderpath, label_file.replace('.txt', f'.{ext}'))
            if os.path.exists(temp_image_path):
                image_path = temp_image_path
                break
        if not image_path:
            continue  # Skip if no image file found with any of the specified extensions
        # Open the corresponding image
        with Image.open(image_path) as img:
            # Initialize drawing tool
            draw = ImageDraw.Draw(img)
            # Open the label file and read its lines
            with open(os.path.join(labels_folderpath, label_file), 'r') as f:
                lines = f.readlines()
                # Iterate over each line in the label file
                for line in lines:
                    parts = line.split()
                    class_id, x, y, w, h = map(float, parts[:5])
                    # Convert normalized coordinates to absolute coordinates using the image dimensions
                    x, y, w, h = x * img.size[0], y * img.size[1], w * img.size[0], h * img.size[1]
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    # Draw the bounding box on the image with a thicker outline and a brighter shade of green
                    draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)  # Changed outline color to a brighter green and increased outline width to 5
            # Save the annotated image
            # Ensure the file extension is correct before saving
            image_name = label_file.replace('.txt', '')
            for ext in image_extensions:
                if image_path.endswith(ext):
                    annotated_image_path = os.path.join(annotated_images_dir, f"{image_name}.{ext}")
                    img.save(annotated_image_path)
                    break  # Exit the loop once the image is saved

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Choose operations to perform on YOLO format data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the parent directory containing both labels and images folders")
    parser.add_argument("--convert_to_coco", action="store_true", help="Convert YOLO format to COCO format")
    parser.add_argument("--draw_annotations", action="store_true", help="Draw annotations on images and save them in a new directory")
    parser.add_argument("--create_csv", action="store_true", help="Create a CSV file with bounding box information")
    # Parse the arguments
    args = parser.parse_args()

    # Perform operations based on arguments
    if args.convert_to_coco:
        convert_to_coco_format(args.data_dir)
    if args.draw_annotations:
        draw_annotations(args.data_dir)
    if args.create_csv:
        # Assuming there's a function to create a CSV file
        create_csv(args.data_dir)