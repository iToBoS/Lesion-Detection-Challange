#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""YOLO Format Object Detection Evaluation Tool.

This module provides functionality to evaluate object detection models that use
YOLO format annotations. It calculates standard metrics including precision,
recall, accuracy, and F1 score by comparing predicted bounding boxes against
ground truth annotations.

Requirements:
------------
- Python 3.6+
- NumPy >= 1.19.0

Directory Structure:
------------------
The tool requires three directories (can be in any location):

1. Images Directory:
   - Contains the image files
   - Example: /path/to/dataset/images/
     ├── image1.jpg
     ├── image2.jpg
     └── ...

2. Ground Truth Directory:
   - Contains YOLO format annotation files
   - Files must have same base names as corresponding images
   - Example: /path/to/ground_truth/labels/
     ├── image1.txt
     ├── image2.txt
     └── ...

3. Predictions Directory:
   - Contains YOLO format prediction files
   - Files must have same base names as corresponding images
   - Example: /path/to/predictions/labels/
     ├── image1.txt
     ├── image2.txt
     └── ...

Note: The three directories can be completely independent of each other
      as long as the annotation files match image base names.

Input Formats:
------------
1. Image Files:
   - Supported formats: jpg, jpeg, png
   - Must have corresponding annotation files with same base name

2. Annotation Files (both ground truth and predictions):
   - YOLO format text files (.txt)
   - One box per line: <class_id> <x_center> <y_center> <width> <height>
   - All values are normalized to [0, 1]
   - Example:
     0 0.5 0.5 0.2 0.3
     1 0.7 0.8 0.1 0.2

Output Control:
-------------
1. Verbose Mode (--verbose):
   - Displays detailed progress during evaluation
   - Shows per-image detection counts
   - Default: False

2. Save Results (--save_results):
   - Creates evaluation_report.txt with comprehensive results
   - Default output directory: './evaluation_results/'
   - Can be customized with --output_dir
   - Report includes:
     * Summary metrics
     * Detailed lists of TP, FP, FN images
     * Per-category counts
     * Path to images with TP, FP, FN

Usage Examples:
-------------
1. Basic usage:
   ```python
   from yolo_evaluation import evaluate_dataset
   
   results = evaluate_dataset(
       images_dir='/path/to/dataset/images',
       gt_dir='/path/to/ground_truth/labels',
       pred_dir='/path/to/predictions/labels',
       iou_threshold=0.5,
       verbose=True,
       save_results=True,
       output_dir='my_results'
   )
   ```

2. Command line usage:
   ```bash
   python yolo_evaluation.py \
       --images_dir /path/to/dataset/images \
       --gt_dir /path/to/ground_truth/labels \
       --pred_dir /path/to/predictions/labels \
       --iou_threshold 0.5 \
       --verbose \
       --save_results \
       --output_dir my_results
   ```

Returns:
-------
Dictionary with metrics:
    {
        'precision': float,  # TP / (TP + FP)
        'recall': float,     # TP / (TP + FN)
        'accuracy': float,   # TP / (TP + FP + FN)
        'f1_score': float,   # 2 * (precision * recall) / (precision + recall)
        'true_positives': int,
        'false_positives': int,
        'false_negatives': int
    }

Notes:
-----
- IoU (Intersection over Union) is used to determine matches between
  predicted and ground truth boxes
- Default IoU threshold is 0.5
- Only boxes of the same class are compared
- Results are saved to './evaluation_results/' by default if save_results=True

Author: Joseph Adeola
License: MIT
"""

import os
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Set, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_yolo_label(label_path: Path) -> List[List[float]]:
    """Read YOLO format annotation file and extract bounding box information.
    
    Args:
        label_path: Path to the YOLO format label file
        
    Returns:
        List of bounding boxes, where each box is [class_id, x, y, w, h]
    """
    if not os.path.exists(label_path):
        logger.info(f"Label file {label_path} does not exist.")
        return []
    elif os.path.getsize(label_path) == 0:
        logger.info(f"Label file {label_path} is empty.")
        return []
        
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x, y, w, h = map(float, line.strip().split())
            boxes.append([class_id, x, y, w, h])
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Read {len(boxes)} boxes from {label_path}.")
    return boxes

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min = box1[1] - box1[3]/2
    x1_max = box1[1] + box1[3]/2
    y1_min = box1[2] - box1[4]/2
    y1_max = box1[2] + box1[4]/2
    
    x2_min = box2[1] - box2[3]/2
    x2_max = box2[1] + box2[3]/2
    y2_min = box2[2] - box2[4]/2
    y2_max = box2[2] + box2[4]/2
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1[3] * box1[4]
    box2_area = box2[3] * box2[4]
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def evaluate_image(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_threshold: float
) -> Tuple[int, int, int]:
    """Evaluate predictions for a single image."""
    if not gt_boxes and not pred_boxes:
        if logger.isEnabledFor(logging.INFO):
            logger.info("No ground truth or prediction boxes.")
        return 0, 0, 0
    
    if not gt_boxes:
        if logger.isEnabledFor(logging.INFO):
            logger.info("No ground truth boxes.")
        return 0, len(pred_boxes), 0
    
    if not pred_boxes:
        if logger.isEnabledFor(logging.INFO):
            logger.info("No prediction boxes.")
        return 0, 0, len(gt_boxes)

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            if gt_box[0] == pred_box[0]:  # Same class
                iou_matrix[i, j] = calculate_iou(gt_box, pred_box)

    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()
    true_positives = 0

    flat_indices = np.argsort(iou_matrix.ravel())[::-1]
    for idx in flat_indices:
        i, j = np.unravel_index(idx, iou_matrix.shape)
        
        if (i not in matched_gt and 
            j not in matched_pred and 
            iou_matrix[i, j] >= iou_threshold):
            
            true_positives += 1
            matched_gt.add(i)
            matched_pred.add(j)

    false_positives = len(pred_boxes) - true_positives
    false_negatives = len(gt_boxes) - true_positives
    
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Image evaluation: TP={true_positives}, FP={false_positives}, FN={false_negatives}")
    return true_positives, false_positives, false_negatives

def save_evaluation_results(
    results_dir: Path,
    metrics: Dict[str, float],
    results: Dict[str, List[str]]
) -> None:
    """Save evaluation results to a detailed report file."""
    report_file = results_dir / "evaluation_report.txt"
    
    with report_file.open('w') as f:
        # Write header with timestamp
        f.write(f"YOLO Detection Evaluation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write summary metrics
        f.write("Summary Metrics\n")
        f.write("--------------\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Write detailed results for each category
        for category, paths in results.items():
            if paths:  # Only write non-empty categories
                f.write(f"\n{category.replace('_', ' ').title()}\n")
                f.write("-" * len(category) + "\n")
                f.write(f"Total: {len(paths)}\n\n")
                for path in paths:
                    f.write(f"{path}\n")

def evaluate_dataset(
    images_dir: str,
    gt_dir: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
    verbose: bool = False,
    save_results: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate object detection performance on entire dataset.
    
    Args:
        images_dir: Directory containing images
        gt_dir: Directory containing ground truth annotations
        pred_dir: Directory containing predicted annotations
        iou_threshold: Minimum IoU for a match (default: 0.5)
        verbose: Whether to show detailed progress (default: False)
        save_results: Whether to save evaluation report (default: False)
        output_dir: Directory to save results (default: './evaluation_results')
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    total_tp = total_fp = total_fn = 0
    false_negatives_paths = []
    false_positives_paths = []
    true_positives_paths = []
    
    # Count total images for progress tracking
    images = list(Path(images_dir).glob('*'))
    total_images = len(images)
    
    if verbose:
        logger.info(f"Starting evaluation of {total_images} images...")
    
    for idx, img_path in enumerate(images, 1):
        if verbose:
            logger.info(f"Processing image {idx}/{total_images}: {img_path.name}")
            
        img_name = img_path.stem
        gt_path = Path(gt_dir) / f"{img_name}.txt"
        pred_path = Path(pred_dir) / f"{img_name}.txt"
        
        gt_boxes = read_yolo_label(gt_path)
        pred_boxes = read_yolo_label(pred_path)
        
        tp, fp, fn = evaluate_image(gt_boxes, pred_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Track results by image
        if fn > 0:
            false_negatives_paths.append(str(img_path))
        if fp > 0:
            false_positives_paths.append(str(img_path))
        if tp > 0:
            true_positives_paths.append(str(img_path))

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn
    }
    
    # Save results if requested
    if save_results:
        results_dir = Path(output_dir if output_dir else './evaluation_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'false_negatives': false_negatives_paths,
            'false_positives': false_positives_paths,
            'true_positives': true_positives_paths
        }
        
        save_evaluation_results(results_dir, metrics, results)
        logger.info(f"Evaluation results saved to: {results_dir}/evaluation_report.txt")
    
    if verbose:
        logger.info("\nEvaluation Summary:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1_score:.4f}")
        logger.info(f"True Positives: {total_tp}")
        logger.info(f"False Positives: {total_fp}")
        logger.info(f"False Negatives: {total_fn}")
    
    return metrics

def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO format object detection results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory containing the image files"
    )
    parser.add_argument(
        "--gt_dir",
        required=True,
        help="Directory containing ground truth annotation files"
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing prediction annotation files"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching boxes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress during evaluation"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save evaluation results to file"
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Validate directories exist
    for dir_path in [args.images_dir, args.gt_dir, args.pred_dir]:
        if not os.path.exists(dir_path):
            logger.error(f"Directory does not exist: {dir_path}")
            return

    try:
        results = evaluate_dataset(
            images_dir=args.images_dir,
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            iou_threshold=args.iou_threshold,
            verbose=args.verbose,
            save_results=args.save_results,
            output_dir=args.output_dir
        )

        # Print results with color formatting
        logger.info("\n" + "=" * 50)
        logger.info(f"\033[92mEvaluation Results (IoU threshold: {args.iou_threshold})")
        logger.info("-" * 50)
        logger.info(f"True Positives: {results['true_positives']}")
        logger.info(f"False Positives: {results['false_positives']}")
        logger.info(f"False Negatives: {results['false_negatives']}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}\033[0m")
        logger.info("=" * 50 + "\n")

        if args.save_results:
            logger.info(f"Detailed results saved to: {args.output_dir}/evaluation_report.txt")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
