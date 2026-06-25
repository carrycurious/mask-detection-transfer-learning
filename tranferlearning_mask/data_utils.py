import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes(path_or_list):
    if isinstance(path_or_list, list):
        return [str(item).strip() for item in path_or_list]
    if os.path.isfile(path_or_list):
        with open(path_or_list, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise FileNotFoundError(f"Class list not found: {path_or_list}")


def collate_fn(batch):
    return tuple(zip(*batch))


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    if filename is None or size is None:
        return None

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.findtext("name")
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x_min = float(bndbox.findtext("xmin"))
        y_min = float(bndbox.findtext("ymin"))
        x_max = float(bndbox.findtext("xmax"))
        y_max = float(bndbox.findtext("ymax"))
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
    }


def get_default_transform(resize=None):
    def _transform(image, target):
        if resize is not None:
            image = image.resize(resize, Image.BILINEAR)
            width, height = image.size
            if "boxes" in target:
                scale_x = width / target["orig_width"]
                scale_y = height / target["orig_height"]
                boxes = target["boxes"]
                boxes = [
                    [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
                    for b in boxes
                ]
                target["boxes"] = boxes
        image = F.to_tensor(image)
        return image, target

    return _transform


class MaskDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, transforms=None):
        self.image_dir = os.path.abspath(image_dir)
        self.annotation_dir = os.path.abspath(annotation_dir)
        self.classes = [str(item).strip() for item in classes]
        self.transforms = transforms
        self.samples = []

        xml_files = sorted(glob.glob(os.path.join(self.annotation_dir, "*.xml")))
        for xml_path in xml_files:
            record = parse_voc_xml(xml_path)
            if record is None:
                continue
            image_name = os.path.basename(record["filename"])
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(image_path):
                continue
            self.samples.append({
                "image_path": image_path,
                "annotation": record,
            })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No annotation/image pairs found in {self.annotation_dir} and {self.image_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        annotation = sample["annotation"]

        boxes = annotation["boxes"]
        labels = [self.classes.index(label) + 1 for label in annotation["labels"]]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.as_tensor(
                [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "orig_width": annotation["width"],
            "orig_height": annotation["height"],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


class YOLOv1Dataset(Dataset):
    def __init__(self, image_dir, annotation_dir, classes, S=7, B=2, resize=(448, 448)):
        self.inner = MaskDetectionDataset(image_dir, annotation_dir, classes)
        self.S = S
        self.B = B
        self.C = len(classes)
        self.resize = resize

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        image, target = self.inner[idx]
        image = F.to_pil_image(image)
        image = image.resize(self.resize, Image.BILINEAR)
        image = F.to_tensor(image)
        target_tensor = self._encode_target(target, self.resize)
        return image, target_tensor

    def _encode_target(self, target, resize):
        boxes = target["boxes"].clone()
        labels = target["labels"].clone()
        _, _, width, height = 3, *resize
        boxes = boxes / torch.tensor([target["orig_width"], target["orig_height"], target["orig_width"], target["orig_height"]], dtype=torch.float32)
        target_tensor = torch.zeros((self.S, self.S, self.C + 5 * self.B), dtype=torch.float32)

        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            box_w = x_max - x_min
            box_h = y_max - y_min
            col = min(self.S - 1, int(x_center * self.S))
            row = min(self.S - 1, int(y_center * self.S))
            x_cell = x_center * self.S - col
            y_cell = y_center * self.S - row

            for b in range(self.B):
                start = 5 * b
                target_tensor[row, col, start : start + 4] = torch.tensor([x_cell, y_cell, box_w, box_h])
                target_tensor[row, col, start + 4] = 1.0
            class_offset = 5 * self.B
            target_tensor[row, col, class_offset + label - 1] = 1.0

        return target_tensor
