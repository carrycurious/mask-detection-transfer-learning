# YoloMaskdetection

This repository contains a transfer learning and fine-tuning workflow for real-time mask detection using YOLOv8, YOLOv4, Faster R-CNN, and an enhanced YOLOv1 architecture.

## Getting Started

Open the `tranferlearning_mask` directory and follow the README there:

```bash
cd tranferlearning_mask
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Then use the training, evaluation, and detection scripts inside the folder.

## Core Features

- Transfer learning on YOLOv8 with Ultralytics
- YOLOv4 Darknet dataset preparation and inference support
- Faster R-CNN training with torchvision
- Enhanced YOLOv1 model with regularization and batch normalization
- Metrics reporting for mAP, IoU, and F2-score
