#!/usr/bin/env python3
"""
YOLO Annotation Visualization Tool
----------------------------------

A tool for visualizing YOLO format annotations on images. This script displays original images
alongside their annotated versions with bounding boxes drawn according to YOLO format labels. 


Example Usage:
--------------
   python visualize_annotation.py --image-dir "./my_images" --label-dir "./my_labels" --num-files 10 --scale 0.75

   Parameters:
    --image-dir      Directory containing your images (.png, .jpg, .jpeg)
    --label-dir      Directory containing your YOLO label files (.txt)
    --num-files 10   Number of random images to show (default: 25)
                    Examples: --num-files 5 (show 5 images), --num-files 50 (show 50 images)
    --scale 0.75     Display size multiplier (default: 0.5)
                    Examples: --scale 0.5 (half size), --scale 1.0 (original size), --scale 2.0 (double size)

Controls:
--------
- Press ESC to exit
- Press any other key to move to next image


Input Directory Structure:
--------------------------
    my_images/
    ├── image1.png
    ├── image2.png
    └── ...

    my_labels/
    ├── image1.txt
    ├── image2.txt
    └── ...

Label Format (YOLO):
---------------------------
Each line in the label file should contain: <class_id> <x_center> <y_center> <width> <height>
where all values are normalized between 0 and 1.


"""


import os
import random
import argparse
import cv2
import numpy as np
from typing import List, Tuple


class YOLOVisualizer:
    """Class for handling YOLO format annotation visualization."""
    
    def __init__(self, image_dir: str, label_dir: str, scale_factor: float = 0.5):
        """
        Initialize the YOLO visualizer.

        Args:
            image_dir (str): Path to directory containing images
            label_dir (str): Path to directory containing YOLO format labels
            scale_factor (float): Scale factor for display size (0-1)
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.scale_factor = scale_factor
        self.supported_formats = (".png", ".jpg", ".jpeg")

    def draw_yolo_annotations(self, image: np.ndarray, 
                            annotations: List[Tuple[int, float, float, float, float]]) -> np.ndarray:
        """
        Draw bounding boxes on image from YOLO format annotations.

        Args:
            image (np.ndarray): Input image
            annotations (List[Tuple]): List of YOLO annotations (class_id, x_center, y_center, w, h)

        Returns:
            np.ndarray: Image with drawn annotations
        """
        height, width = image.shape[:2]
        annotated_image = image.copy()
        
        for class_id, x_center, y_center, w, h in annotations:
            # Convert normalized YOLO coordinates to pixel coordinates
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add class ID label
            cv2.putText(annotated_image, f"Class {class_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_image

    def load_annotations(self, label_path: str) -> List[Tuple[int, float, float, float, float]]:
        """
        Load YOLO format annotations from file.

        Args:
            label_path (str): Path to label file

        Returns:
            List[Tuple]: List of annotations (class_id, x_center, y_center, w, h)
        """
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        annotation = (int(parts[0]), *(float(x) for x in parts[1:]))
                        annotations.append(annotation)
        return annotations

    def visualize_dataset(self, num_images: int = 25):
        """
        Visualize random samples from the dataset.

        Args:
            num_images (int): Number of random images to display
        """
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(self.supported_formats)]
        
        if not image_files:
            print(f"No images found in {self.image_dir}")
            return

        selected_images = random.sample(image_files, min(len(image_files), num_images))

        for image_file in selected_images:
            # Load and process image
            image_path = os.path.join(self.image_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            # Load annotations
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(self.label_dir, label_file)
            annotations = self.load_annotations(label_path)

            # Create annotated version
            annotated_image = self.draw_yolo_annotations(image, annotations)

            # Combine and display images
            combined_image = np.hstack((image, annotated_image))
            combined_image = cv2.resize(combined_image, 
                                      (int(combined_image.shape[1] * self.scale_factor),
                                       int(combined_image.shape[0] * self.scale_factor)))

            # Add labels
            cv2.putText(combined_image, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_image, "Annotated", (int(image.shape[1] * self.scale_factor) + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display
            cv2.imshow("YOLO Annotations Viewer", combined_image)
            
            # Handle keyboard input
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                break

        cv2.destroyAllWindows()


def main():
    """Main function to handle command line arguments and run the visualizer."""
    parser = argparse.ArgumentParser(description="Visualize YOLO format annotations on images")
    parser.add_argument("--image-dir", required=True, help="Directory containing images")
    parser.add_argument("--label-dir", required=True, help="Directory containing YOLO labels")
    parser.add_argument("--num-images", type=int, default=25, help="Number of images to display")
    parser.add_argument("--scale", type=float, default=0.8, help="Scale factor for display size")
    
    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.image_dir):
        raise ValueError(f"Image directory not found: {args.image_dir}")
    if not os.path.isdir(args.label_dir):
        raise ValueError(f"Label directory not found: {args.label_dir}")

    # Initialize and run visualizer
    visualizer = YOLOVisualizer(args.image_dir, args.label_dir, args.scale)
    visualizer.visualize_dataset(args.num_images)


if __name__ == "__main__":
    main()