import os
import shutil
import subprocess
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data_utils import (
    YOLOv1Dataset,
    MaskDetectionDataset,
    collate_fn,
    get_default_transform,
    load_yaml,
    load_classes,
)


def _resolve_path(base, path):
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base, path))


def parse_dataset_config(data_yaml):
    data = load_yaml(data_yaml)
    root = os.path.dirname(os.path.abspath(data_yaml))
    train_dir = _resolve_path(root, data["train"])
    val_dir = _resolve_path(root, data["val"])
    classes = load_classes(data.get("names", data.get("classes", [])))
    annotation_dir = _resolve_path(root, "annotations")
    return train_dir, val_dir, annotation_dir, classes


def train_yolov8(
    data_yaml,
    model_name="yolov8n.pt",
    epochs=30,
    batch=16,
    imgsz=640,
    project="runs/train",
    name="yolov8_mask",
):
    model = YOLO(model_name)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
    )
    return os.path.join(project, name, "weights", "best.pt")


def train_fasterrcnn(
    data_yaml,
    epochs=12,
    batch_size=2,
    lr=0.0025,
    output_path="runs/rcnn/fasterrcnn_mask.pth",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir, val_dir, annotation_dir, classes = parse_dataset_config(data_yaml)
    train_dataset = MaskDetectionDataset(
        image_dir=train_dir,
        annotation_dir=annotation_dir,
        classes=classes,
        transforms=get_default_transform(resize=(800, 800)),
    )
    val_dataset = MaskDetectionDataset(
        image_dir=val_dir,
        annotation_dir=annotation_dir,
        classes=classes,
        transforms=get_default_transform(resize=(800, 800)),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - loss={epoch_loss/len(train_loader):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


class TinyYOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=3):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.conv = nn.Sequential(
            self._block(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            self._block(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            self._block(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            self._block(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            self._block(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            self._block(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + 5 * B)),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.C + 5 * self.B)
        return x


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=3, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        coord_loss = 0.0
        obj_loss = 0.0
        noobj_loss = 0.0
        class_loss = 0.0

        for pred, target in zip(predictions, targets):
            obj_mask = target[..., 4] > 0
            noobj_mask = target[..., 4] == 0
            coord_loss += torch.sum((pred[..., :4][obj_mask] - target[..., :4][obj_mask]) ** 2)
            obj_loss += torch.sum((pred[..., 4][obj_mask] - target[..., 4][obj_mask]) ** 2)
            noobj_loss += torch.sum((pred[..., 4][noobj_mask] - target[..., 4][noobj_mask]) ** 2)
            class_loss += torch.sum((pred[..., 5:] - target[..., 5:]) ** 2)

        loss = (
            self.lambda_coord * coord_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )
        return loss / batch_size


def train_yolo1(
    data_yaml,
    epochs=20,
    batch_size=8,
    lr=1e-4,
    output_path="runs/yolo1/yolo1_mask.pth",
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir, _, annotation_dir, classes = parse_dataset_config(data_yaml)
    dataset = YOLOv1Dataset(
        image_dir=train_dir,
        annotation_dir=annotation_dir,
        classes=classes,
        resize=(448, 448),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = TinyYOLOv1(S=7, B=2, C=len(classes)).to(device)
    criterion = YOLOv1Loss(S=7, B=2, C=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"YOLOv1 Epoch {epoch+1}/{epochs} - loss={epoch_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


def build_darknet_data(data_yaml, output_dir="runs/darknet"):
    train_dir, val_dir, annotation_dir, classes = parse_dataset_config(data_yaml)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    names_path = os.path.join(output_dir, "mask.names")
    data_path = os.path.join(output_dir, "mask.data")
    train_list = os.path.join(output_dir, "train.txt")
    val_list = os.path.join(output_dir, "val.txt")

    with open(names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(classes).strip() + "\n")

    with open(data_path, "w", encoding="utf-8") as f:
        f.write(f"classes = {len(classes)}\n")
        f.write(f"train = {train_list}\n")
        f.write(f"valid = {val_list}\n")
        f.write(f"names = {names_path}\n")
        f.write(f"backup = {output_dir}\n")

    def dump_list(images_dir, output_file):
        with open(output_file, "w", encoding="utf-8") as writer:
            for image_name in sorted(os.listdir(images_dir)):
                if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    abs_path = os.path.abspath(os.path.join(images_dir, image_name))
                    writer.write(abs_path + "\n")

    dump_list(train_dir, train_list)
    dump_list(val_dir, val_list)
    return data_path, names_path, train_list, val_list


def train_yolov4(data_yaml, cfg_path, weights_path, darknet_executable="darknet", output_dir="runs/darknet"):
    data_path, names_path, train_list, val_list = build_darknet_data(data_yaml, output_dir)
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("YOLOv4 cfg or weights not found. Download the official Darknet YOLOv4 files and provide the correct paths.")

    command = [
        darknet_executable,
        "detector",
        "train",
        data_path,
        cfg_path,
        weights_path,
        "-dont_show",
    ]
    print("Running Darknet command:", " ".join(command))
    subprocess.run(command, check=True)
    return output_dir


def detect_yolov4_image(image_path, cfg_path, weights_path, names_path, conf_threshold=0.5):
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_layer_names = net.getUnconnectedOutLayersNames()
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layer_names)
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(scores.argmax())
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)
    return boxes, confidences, class_ids
