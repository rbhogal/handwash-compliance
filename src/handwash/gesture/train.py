"""
Training script for the Layer 2 YOLOv8-classify gesture model.
Fine-tunes a pretrained YOLOv8n-cls on the WHO 7-step gesture dataset.

Run:
    # 1. Prepare data (once)
    python -c "
    from handwash.gesture.dataset import prepare_yolo_dataset
    prepare_yolo_dataset('data/raw/gestures', 'data/processed/gestures')
    "

    # 2. Train
    python src/handwash/gesture/train.py

    # Or with overrides:
    python src/handwash/gesture/train.py --epochs 60 --weights yolov8s-cls.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 gesture classifier")
    p.add_argument(
        "--data",
        default="data/processed/gestures",
        help="Root of the classify dataset (contains train/ val/ test/)",
    )
    p.add_argument(
        "--weights",
        default="yolov8n-cls.pt",
        help="Starting weights — yolov8n-cls.pt / yolov8s-cls.pt / yolov8m-cls.pt",
    )
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--imgsz",   type=int,   default=224)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr0",     type=float, default=1e-3)
    p.add_argument("--device",  default="0" if _gpu_available() else "cpu")
    p.add_argument("--project", default="runs/gesture",  help="Save directory")
    p.add_argument("--name",    default="exp",           help="Run name")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run prepare_yolo_dataset() first (see module docstring)."
        )

    model = YOLO(args.weights)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        project=args.project,
        name=args.name,
        # Regularisation / augmentation
        dropout=0.2,
        flipud=0.0,    # vertical flip not meaningful for hand gestures
        fliplr=0.5,
        degrees=15.0,  # slight rotation is realistic
        translate=0.1,
        scale=0.2,
    )

    # Copy best weights to canonical location
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    dest = Path("models/weights/gesture_classifier.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        import shutil
        shutil.copy2(best_weights, dest)
        print(f"\nBest weights saved to {dest}")

    print(f"Training complete. Full results in {args.project}/{args.name}")


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
