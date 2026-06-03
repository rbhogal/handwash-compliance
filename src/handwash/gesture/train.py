"""
Training script for the Layer 2 YOLOv8-classify gesture model.
Fine-tunes a pretrained YOLOv8n-cls on the METC WHO gesture dataset.

Run:
    # 1. Prepare data (once)
    python scripts/prepare_gesture_data.py

    # 2. Train
    python src/handwash/gesture/train.py

    # Or with overrides:
    python src/handwash/gesture/train.py --epochs 60 --weights yolov8s-cls.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 gesture classifier")
    p.add_argument(
        "--data",
        default="data/gesture/dataset",
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
    p.add_argument("--device",  default=_default_device())
    p.add_argument("--project", default="runs/gesture", help="Save directory — ultralytics saves to {project}/{name}/weights/")
    p.add_argument("--name",    default="exp",          help="Run name")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run scripts/prepare_gesture_data.py first."
        )

    print(f"Starting weights : {args.weights}")
    print(f"Dataset          : {data_path}")
    print(f"Epochs           : {args.epochs}")
    print(f"Image size       : {args.imgsz}")
    print(f"Batch size       : {args.batch}")
    print(f"Device           : {args.device}")
    print()

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
        dropout=0.2,
        flipud=0.0,    # vertical flip not meaningful for hand gestures
        fliplr=0.0,    # horizontal flip disabled — mirroring creates false gesture variants
        degrees=15.0,  # slight rotation is realistic
        translate=0.1,
        scale=0.2,
    )

    best_weights = results.save_dir / "weights" / "best.pt"
    dest = Path("models/weights/gesture_classifier.pt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        shutil.copy2(best_weights, dest)
        print(f"\nBest weights saved to {dest}")

    print(f"Training complete. Full results in {results.save_dir}")


def _default_device() -> str:
    """Return mps on Apple Silicon, cuda if available, else cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


if __name__ == "__main__":
    main()
