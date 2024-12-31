<!-- <img src="images/logo.png" alt="Logo" width="200"> # iToBoS 2024 - Skin Lesion Detection with 3D-TBP -->
<div style="display: flex; align-items: center;">
    <img src="images/itoboslogo.png" alt="Logo" width="150" style="margin-right: 20px;">
      <h1 style="font-weight: bold; margin-bottom: 0;">2024 - Skin Lesion Detection with 3D-TBP</h1>
    <!-- # iToBoS 2024 - Skin Lesion Detection with 3D-TBP -->
</div>



[![Challenge Page](https://img.shields.io/badge/Kaggle-blue.svg)](https://www.kaggle.com/competitions/itobos-2024-detection)
[![Paper](https://img.shields.io/badge/Paper-purple.svg)](YOUR_PAPER_URL_HERE)
<!-- Replace YOUR_PAPER_URL_HERE with the actual link to the PDF of your paper -->

Official repository for the iToBoS skin lesion detection challenge dataset utilizing Total Body Photography (TBP) imaging.

## Overview

The iToBoS challenge provides a comprehensive benchmark for evaluating skin lesion detection algorithms using advanced 3D Total Body Photography. This repository contains the dataset, evaluation tools, and helper scripts necessary for training and testing models.

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Samples from Dataset](#samples-from-dataset)
- [Helper Scripts](#helper-scripts)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- Complete skin lesion detection dataset with annotations on Kaggle [here](https://www.kaggle.com/competitions/itobos-2024-detection/data)
- YOLO and COCO format support
- Conversion utilities between annotation formats
- Visualization tools for annotations
- Statistical analysis notebooks
- Evaluation scripts for model performance

## Directory Structure
<!-- This is a comment
├── dataset/                     # Dataset root directory
│   ├── train/                  # Training dataset
│   │   ├── images/            # Training images
│   │   └── labels/            # Training annotations
│   ├── test/                  # Test dataset
│   │   ├── images/            # Test images
│   │   └── labels/            # Test annotations
│   └── additional_data/       # Supplementary data
│
-->
```
iToBoS-Challenge/
├── helpers/                    # Utility scripts
│   ├── coco_to_yolo.py        # COCO to YOLO format converter
│   ├── yolo_to_coco.py        # YOLO to COCO format converter
│   ├── statistics.ipynb       # Dataset analysis notebook
│   ├── visualize_annotation.py # Annotation visualization tool
│   └── yolo_eval.py           # YOLO evaluation script
│
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── LICENSE                    # License information
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iToBoS-Challenge.git
cd iToBoS-Challenge
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset is organized into training and testing sets, each containing:
- High-resolution skin images
- Lesion annotations in YOLO format
- Additional metadata about imaging conditions

### Dataset Statistics
- Training images: 8000
- Test images: 9000
- Lesion classes: [0 : unlabeled]


## Samples from Dataset
<img src="images/itobos-challenge-image-samples" alt="Examples of images in the iToBoS challenge dataset" width="1000">

<!-- Add logo here:-->


## Helper Scripts

The `helpers` directory contains various utility scripts:

1. **Format Conversion**
   - `coco_to_yolo.py`: Converts COCO format annotations to YOLO
   - `yolo_to_coco.py`: Converts YOLO format annotations to COCO

2. **Analysis Tools**
   - `statistics.ipynb`: Jupyter notebook for dataset analysis
   - `visualize_annotation.py`: Tool for visualizing annotations
   - `yolo_eval.py`: Evaluation script for YOLO format predictions

## Usage Examples

### Converting Annotations
```bash
# YOLO to COCO conversion
python helpers/yolo_to_coco.py --input path/to/yolo --output path/to/coco

# COCO to YOLO conversion
python helpers/coco_to_yolo.py --input path/to/coco --output path/to/yolo
```

### Visualizing Annotations
```bash
python helpers/visualize_annotation.py --image path/to/image --label path/to/label
```

### Running Evaluation
```bash
python helpers/yolo_eval.py --pred path/to/predictions --gt path/to/ground_truth
```

<!-- ## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request -->

## License

This project is licensed under the Creative Common License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
This competition was hosted as part of the European Union's Horizon 2020 research and innovation programme iToBoS - Intelligent Total Body Scanner for Early Detection of Melanoma, under grant agreement no. 965221.

## Citation

Josep Malvehy, Peter Soyer, Nuria Ferrera, Clare Primiero, Serena Bonin, Gisele Rezze, Brian D’Alessandro, Anup Saha, Joseph Adeola, Hayat Rajani, Rafael Garcia. iToBoS 2024 - Skin Lesion Detection with 3D-TBP. https://kaggle.com/competitions/itobos-2024-detection, 2024. Kaggle.

