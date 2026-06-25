# Mask Detection Transfer Learning Repository

This repository implements a complete image-based mask detection pipeline using transfer learning and fine-tuning.
It compares modern detectors (YOLOv8, YOLOv4, Faster R-CNN) with an enhanced YOLOv1-style baseline.

## Project Theory

- Object detection locates objects in images and classifies them with bounding boxes.
- Transfer learning reuses pre-trained networks so the model learns faster and generalizes better with limited data.
- YOLO models are single-stage detectors optimized for speed, making them suitable for real-time mask detection.
- Faster R-CNN is a two-stage detector that often delivers stronger localization accuracy at the cost of speed.
- YOLOv1 was the first version of the YOLO family; this project reimplements a compact, regularized version to compare older architecture performance with modern detectors.
- Evaluation metrics used in this project:
  - `mAP@0.5` — mean average precision at IoU threshold 0.5
  - `mAP@[0.5:0.95]` — averaged precision across multiple IoU thresholds
  - `IoU` — Intersection over Union for bounding box overlap
  - `F2-score` — a precision / recall metric that weights recall more heavily

## What this project does

- Reads an image dataset and VOC XML annotations.
- Trains YOLOv8 on the mask dataset using Ultralytics.
- Prepares YOLOv4 Darknet data files and supports inference through a Darknet-style pipeline.
- Fine-tunes a Faster R-CNN detector using PyTorch torchvision.
- Implements a YOLOv1-inspired model with batch normalization and dropout for better regularization.
- Provides evaluation support and live webcam/video inference.

## Dataset details

- The dataset is image-only.
- Training and validation images are expected under `images/train` and `images/val`.
- Annotations are stored in `annotations/` as VOC XML files.
- `dataset.yaml` points to the image splits and defines the class names.

## Repository Structure

- `data_utils.py` - dataset loading, annotation parsing, and YOLOv1 target encoding
- `model_utils.py` - training and inference helpers for YOLOv8, YOLOv4, Faster R-CNN, and YOLOv1
- `train.py` - command-line interface to train any supported model
- `evaluate.py` - validation and mAP-style evaluation for YOLOv8 and Faster R-CNN
- `detect.py` - real-time webcam/video detection utility
- `metrics.py` - IoU, mAP, and F2-score helper functions
- `dataset.yaml` - dataset configuration for training pipelines
- `classes.txt` - class label file

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. To use YOLOv4, provide Darknet files such as `yolov4.cfg` and `yolov4.weights`.

## Training

Train YOLOv8:

```bash
python train.py --model yolov8 --data dataset.yaml --epochs 30 --batch 16
```

Train Faster R-CNN:

```bash
python train.py --model fasterrcnn --data dataset.yaml --epochs 12 --batch 2
```

Train the YOLOv1-style model:

```bash
python train.py --model yolo1 --data dataset.yaml --epochs 20 --batch 8
```

Run YOLOv4 dataset preparation and launch Darknet training:

```bash
python train.py --model yolov4 --data dataset.yaml --cfg yolov4.cfg --weights yolov4.weights --output runs/darknet
```

## Evaluation

Evaluate YOLOv8:

```bash
python evaluate.py --model yolov8 --weights runs/train/yolov8_mask/weights/best.pt --data dataset.yaml
```

Evaluate Faster R-CNN:

```bash
python evaluate.py --model fasterrcnn --weights runs/rcnn/fasterrcnn_mask.pth --data dataset.yaml
```

## Live Inference

YOLOv8 webcam demo:

```bash
python detect.py --model yolov8 --weights runs/train/yolov8_mask/weights/best.pt --names classes.txt --source 0
```

Faster R-CNN webcam demo:

```bash
python detect.py --model fasterrcnn --weights runs/rcnn/fasterrcnn_mask.pth --names classes.txt --source 0
```

YOLOv4 webcam demo:

```bash
python detect.py --model yolov4 --cfg yolov4.cfg --weights yolov4.weights --names classes.txt --source 0
```

## Notes

- This project is built around image-based mask datasets, not direct video training.
- Live inference supports webcam or video streams using trained weights.
- The YOLOv1 variant includes additional regularization and batch normalization to improve training stability.
- The repo is set up to compare modern YOLO and Faster R-CNN performance on mask detection.
