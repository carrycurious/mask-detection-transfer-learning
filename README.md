# YoloMaskdetection

This repository is a mask detection transfer-learning project built around image-based datasets and real-time inference.
It compares modern and classic object detection architectures while supporting training, evaluation, and live webcam/video demo.

## Project Summary

- Uses image datasets with VOC XML annotations for mask detection.
- Trains YOLOv8 using Ultralytics transfer learning.
- Supports YOLOv4 Darknet-style dataset preparation and inference.
- Fine-tunes Faster R-CNN via PyTorch `torchvision`.
- Implements an enhanced YOLOv1-inspired detector with batch normalization and regularization.
- Compares models using metrics such as mAP@0.5, mAP@[0.5:0.95], IoU, and F2-style scoring.

## Getting Started

Open the `tranferlearning_mask` directory and follow the README there:

```bash
cd tranferlearning_mask
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Core Features

- Transfer learning on YOLOv8 for fast mask detection training.
- YOLOv4 Darknet dataset preparation, configuration support, and inference helper.
- Faster R-CNN fine-tuning with torchvision for strong detection accuracy.
- Enhanced YOLOv1 model implementation with batch normalization and dropout.
- Evaluation helpers for mAP, IoU, and F2-score to compare models.
- Live inference scripts for webcam and video source testing.

## Structure

- `tranferlearning_mask/` contains the full training and inference code.
- `dataset.yaml` configures the dataset paths.
- `annotations/` stores VOC XML labels.
- `images/` stores train and validation image splits.

For more details, open `tranferlearning_mask/README.md`.
