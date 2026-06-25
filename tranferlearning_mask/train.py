import argparse
import os

from model_utils import (
    train_yolov4,
    train_yolov8,
    train_fasterrcnn,
    train_yolo1,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mask detection models with transfer learning and fine-tuning."
    )
    parser.add_argument("--model", choices=["yolov8", "yolov4", "fasterrcnn", "yolo1"], required=True,
                        help="Select the model to train.")
    parser.add_argument("--data", default="dataset.yaml", help="Path to the dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--output", default="runs", help="Output directory for model checkpoints.")
    parser.add_argument("--cfg", default=None, help="YOLOv4 cfg path for Darknet training.")
    parser.add_argument("--weights", default=None, help="YOLOv4 pre-trained weights path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLOv8 training.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.model == "yolov8":
        print("Training YOLOv8 model...")
        model_path = train_yolov8(
            args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.output,
            name="yolov8_mask",
        )
        print(f"YOLOv8 best weights saved at: {model_path}")

    elif args.model == "fasterrcnn":
        print("Training Faster R-CNN model...")
        model_path = os.path.join(args.output, "fasterrcnn_mask.pth")
        train_fasterrcnn(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            output_path=model_path,
        )
        print(f"Faster R-CNN weights saved at: {model_path}")

    elif args.model == "yolo1":
        print("Training YOLOv1-style model...")
        model_path = os.path.join(args.output, "yolo1_mask.pth")
        train_yolo1(
            args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            output_path=model_path,
        )
        print(f"YOLOv1 weights saved at: {model_path}")

    elif args.model == "yolov4":
        if args.cfg is None or args.weights is None:
            raise SystemExit("YOLOv4 training requires --cfg and --weights arguments.")
        print("Preparing Darknet files for YOLOv4 training...")
        train_yolov4(
            args.data,
            cfg_path=args.cfg,
            weights_path=args.weights,
            output_dir=args.output,
        )
        print("Darknet YOLOv4 training launched.")


if __name__ == "__main__":
    main()
