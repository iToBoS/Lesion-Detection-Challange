# iToBoS 2024 - Skin Lesion Detection with 3D-TBP
Official repository of the iToBoS skin lesion detection challenge dataset.

## Overview
The iToBoS challenge aims to provide a benchmark for evaluating object detection algorithms. This repository contains the dataset and tools necessary for training and testing models.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure
iToBoS-Challenge/
│
├── dataset/                # Directory for dataset files
│   ├── train/              # Training images and annotations
│   ├── test/               # Test images and annotations
│   └── additional_data/     # Any additional resources
│
├── helpers/                # Helper scripts
│   └── yolo_to_coco.py     # Script to convert YOLO to COCO format
│
├── LICENSE                 # License file
├── requirements.txt        # Required packages
└── README.md               # Project documentation

## Installation
To get started, clone the repository and install the required packages:

```
git clone https://github.com/yourusername/iToBoS-Challenge.git
cd iToBoS-Challenge
pip install -r requirements.txt
```

## Usage
To convert YOLO format annotations to COCO format, use the provided script:

```
python helpers/yolo_to_coco.py --input <path_to_yolo_annotations> --output <path_to_coco_annotations>
```

## Dataset
The dataset is organized into three main directories:
- `train`: Contains training images and annotations.
- `test`: Contains test images and annotations.
- `additional_data`: Any additional resources related to the dataset.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
