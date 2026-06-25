import argparse
import cv2
import torch
from ultralytics import YOLO

from data_utils import load_classes
from model_utils import detect_yolov4_image
from model_utils import parse_dataset_config
from model_utils import train_fasterrcnn


def draw_boxes(image, boxes, labels, confidences, class_names):
    for box, label, conf in zip(boxes, labels, confidences):
        x, y, w, h = box
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{class_names[label]}: {conf:.2f}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def run_yolov8(model_path, class_names, source=0):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Live Mask Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_fasterrcnn(model_path, class_names, source=0, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("pytorch/vision:v0.15.2", "fasterrcnn_resnet50_fpn", pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, len(class_names) + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)[0]
        boxes = output["boxes"].cpu().numpy().astype(int)
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        filtered = [(b, l, s) for b, l, s in zip(boxes, labels, scores) if s >= threshold]
        if filtered:
            frame = draw_boxes(frame, [list(b) for b, _, _ in filtered], [int(l - 1) for _, l, _ in filtered], [float(s) for _, _, s in filtered], class_names)
        cv2.imshow("Faster R-CNN Live Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_yolov4(model_cfg, model_weights, names_path, source=0, threshold=0.5):
    class_names = load_classes(names_path)
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        tmp_image = "tmp_live.jpg"
        cv2.imwrite(tmp_image, frame)
        boxes, confidences, class_ids = detect_yolov4_image(tmp_image, model_cfg, model_weights, names_path, conf_threshold=threshold)
        annotated = draw_boxes(frame, boxes, class_ids, confidences, class_names)
        cv2.imshow("YOLOv4 Live Mask Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run live mask detection from webcam or video source.")
    parser.add_argument("--model", choices=["yolov8", "fasterrcnn", "yolov4"], required=True)
    parser.add_argument("--weights", help="Model weights file path.")
    parser.add_argument("--cfg", help="YOLOv4 cfg file path.")
    parser.add_argument("--names", default="classes.txt", help="Class labels file path.")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam or path to video).")
    args = parser.parse_args()

    if isinstance(args.source, str) and args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    class_names = load_classes(args.names)

    if args.model == "yolov8":
        run_yolov8(args.weights, class_names, source=source)
    elif args.model == "fasterrcnn":
        run_fasterrcnn(args.weights, class_names, source=source)
    else:
        if args.cfg is None or args.weights is None:
            raise SystemExit("YOLOv4 live detection requires --cfg and --weights.")
        run_yolov4(args.cfg, args.weights, args.names, source=source)


if __name__ == "__main__":
    main()
