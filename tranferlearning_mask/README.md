# Mask Detection Transfer Learning Repository

This repository contains a complete workflow for transfer learning and fine-tuning mask detection models using:

- YOLOv8
- YOLOv4 (Darknet integration and inference)
- Faster R-CNN (PyTorch torchvision)
- A YOLOv1-style model enhanced with batch normalization and regularization

## Repository Structure

- `data_utils.py` - dataset loading, annotation parsing and YOLOv1 target encoding
- `model_utils.py` - model training utilities for YOLOv8, YOLOv4, Faster R-CNN, and YOLOv1
- `train.py` - CLI entry point for training models
- `evaluate.py` - evaluation helper for YOLOv8 and Faster R-CNN
- `detect.py` - real-time video detection CLI
- `dataset.yaml` - dataset configuration for Ultralytics and training pipelines
- `classes.txt` - class labels used by the detector

## Setup

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Install Darknet or YOLOv4 weights if you want YOLOv4 transfer learning.

## Training

Train YOLOv8:

```bash
python train.py --model yolov8 --data dataset.yaml --epochs 30 --batch 16
```

Train Faster R-CNN:

```bash
python train.py --model fasterrcnn --data dataset.yaml --epochs 12 --batch 2
```

Train YOLOv1-style model:

```bash
python train.py --model yolo1 --data dataset.yaml --epochs 20 --batch 8
```

Run YOLOv4 Darknet fine-tuning:

```bash
python train.py --model yolov4 --data dataset.yaml --cfg yolov4.cfg --weights yolov4.weights --output runs/darknet
```

## Evaluation

Evaluate a YOLOv8 model:

```bash
python evaluate.py --model yolov8 --weights runs/train/yolov8_mask/weights/best.pt --data dataset.yaml
```

Evaluate a Faster R-CNN model:

```bash
python evaluate.py --model fasterrcnn --weights runs/rcnn/fasterrcnn_mask.pth --data dataset.yaml
```

## Live Inference

Live webcam detection with YOLOv8:

```bash
python detect.py --model yolov8 --weights runs/train/yolov8_mask/weights/best.pt --names classes.txt --source 0
```

Live webcam detection with Faster R-CNN:

```bash
python detect.py --model fasterrcnn --weights runs/rcnn/fasterrcnn_mask.pth --names classes.txt --source 0
```

Live webcam detection with YOLOv4:

```bash
python detect.py --model yolov4 --cfg yolov4.cfg --weights yolov4.weights --names classes.txt --source 0
```

## Notes

- The repository provides a complete pipeline for training, evaluation, and real-time inference.
- `dataset.yaml` references `images/train` and `images/val` directories and expects `annotations/` to contain VOC XML annotations.
- The YOLOv1 model is implemented with additional batch normalization and regularization layers to improve stability.
