# YoloMaskdetection

A full transfer-learning and fine-tuning repository for real-time face mask detection.

This project is built to compare multiple object detection architectures using an image-only mask dataset and VOC annotation files. It supports training, evaluation, and live inference while highlighting model performance trade-offs in speed and accuracy.

## Project Description

The repository is designed around the following objectives:

- Apply transfer learning to YOLOv8 for fast real-time mask detection.
- Support YOLOv4 Darknet-style training and inference for interoperability.
- Fine-tune a Faster R-CNN detector using PyTorch torchvision.
- Implement a custom YOLOv1-style model with batch normalization and regularization.
- Enable live inference on webcam and video sources for real-time demonstration.
- Compare model performance using `mAP@0.5`, `mAP@[0.5:0.95]`, `IoU`, and `F2-score`.

## What this repo contains

- `tranferlearning_mask/` — main project folder with code, dataset config, and annotation support.
- `tranferlearning_mask/data_utils.py` — dataset parsing, annotation loader, and YOLO-target conversion.
- `tranferlearning_mask/model_utils.py` — training utilities for YOLOv8, YOLOv4, Faster R-CNN, and YOLOv1.
- `tranferlearning_mask/train.py` — CLI for training different models from one entrypoint.
- `tranferlearning_mask/evaluate.py` — evaluation scripts for validation and metric reporting.
- `tranferlearning_mask/detect.py` — live webcam/video inference support.
- `tranferlearning_mask/metrics.py` — IoU, mAP, and F2-score helpers.
- `tranferlearning_mask/dataset.yaml` — dataset configuration file.
- `tranferlearning_mask/classes.txt` — label definitions.
- `tranferlearning_mask/annotations/` — VOC XML label files.
- `tranferlearning_mask/images/` — training and validation image splits.

## Key model components

- **YOLOv8** — modern single-stage detector, optimized for real-time speed and easy transfer learning with Ultralytics.
- **YOLOv4** — Darknet-compatible detector, supported using text dataset formats and external YOLOv4 config/weights.
- **Faster R-CNN** — two-stage detector with strong localization and classification performance, useful for accuracy comparison.
- **YOLOv1-style model** — custom compact network with batch normalization and dropout to compare classical detection design against modern models.

## Dataset format

This repo uses an image-only dataset with VOC annotations:

- `images/train` — training images
- `images/val` — validation images
- `annotations/*.xml` — VOC boxes and labels for each image
- `dataset.yaml` — references the train/val folders and class names

## Getting started

1. Enter the project folder:

```bash
cd tranferlearning_mask
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training commands

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

Run YOLOv4 dataset preparation and start Darknet training:

```bash
python train.py --model yolov4 --data dataset.yaml --cfg yolov4.cfg --weights yolov4.weights --output runs/darknet
```

## Evaluation commands

Evaluate YOLOv8:

```bash
python evaluate.py --model yolov8 --weights runs/train/yolov8_mask/weights/best.pt --data dataset.yaml
```

Evaluate Faster R-CNN:

```bash
python evaluate.py --model fasterrcnn --weights runs/rcnn/fasterrcnn_mask.pth --data dataset.yaml
```

## Live inference commands

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

- This repo is built for **image-based** mask detection training.
- Live inference is supported with webcam or video streams.
- YOLOv4 requires external Darknet binaries and config/weights files.
- The YOLOv1-style model is provided as a baseline with added batch normalization and dropout to stabilize training.
- Use the evaluation scripts to compare model performance across accuracy and precision/recall metrics.
