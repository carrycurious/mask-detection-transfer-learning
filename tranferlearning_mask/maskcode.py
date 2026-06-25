import argparse
import subprocess
import sys

import ultralytics


def print_status():
    print("YOLO Mask Detection repository initialized.")
    print(f"Ultralytics version: {ultralytics.__version__}")
    print("Run training with: python train.py --model yolov8 --data dataset.yaml")


def main():
    parser = argparse.ArgumentParser(description="Mask detection utility script.")
    parser.add_argument("--info", action="store_true", help="Print repository status information.")
    parser.add_argument("--train", action="store_true", help="Forward remaining args to train.py.")
    args, remaining = parser.parse_known_args()

    if args.info:
        print_status()
    elif args.train:
        command = [sys.executable, "train.py"] + remaining
        subprocess.run(command)
    else:
        print_status()


if __name__ == "__main__":
    main()
