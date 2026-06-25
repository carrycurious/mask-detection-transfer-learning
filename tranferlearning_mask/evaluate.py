import argparse
import os

import torch
from ultralytics import YOLO

from data_utils import load_yaml, load_classes, MaskDetectionDataset, collate_fn
from metrics import compute_map
from model_utils import parse_dataset_config


def evaluate_yolov8(model_path, data_yaml):
    model = YOLO(model_path)
    print("Validating YOLOv8 on validation dataset...")
    results = model.val(data=data_yaml, imgsz=640, batch=16)
    print(results)
    return results


def evaluate_fasterrcnn(model_path, data_yaml, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, val_dir, annotation_dir, classes = parse_dataset_config(data_yaml)
    val_dataset = MaskDetectionDataset(
        image_dir=val_dir,
        annotation_dir=annotation_dir,
        classes=classes,
        transforms=None,
    )

    model = torch.load(model_path, map_location=device) if model_path.endswith(".pt") else None
    if model is None:
        model = torch.hub.load("pytorch/vision:v0.15.2", "fasterrcnn_resnet50_fpn", pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torch.nn.Linear(in_features, len(classes) + 1)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    ground_truths = []
    for idx, (image, target) in enumerate(val_dataset):
        image = image.to(device)
        with torch.no_grad():
            output = model([image])[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
            predictions.append({
                "image_id": idx,
                "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "score": float(score),
                "class": int(label),
            })

        gts = []
        for box, label in zip(target["boxes"].numpy(), target["labels"].numpy()):
            gts.append({"box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "class": int(label)})
        ground_truths.append(gts)

    map_value, _ = compute_map(predictions, ground_truths)
    print(f"Estimated mAP@[0.5:0.95] for Faster R-CNN: {map_value:.4f}")
    return map_value


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained mask detection model.")
    parser.add_argument("--model", choices=["yolov8", "fasterrcnn"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="dataset.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model == "yolov8":
        evaluate_yolov8(args.weights, args.data)
    else:
        evaluate_fasterrcnn(args.weights, args.data)


if __name__ == "__main__":
    main()
