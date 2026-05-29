"""
Evaluation script — measures gesture classifier accuracy on held-out test split
using Ultralytics' built-in val pipeline.

Usage:
    python scripts/evaluate.py --weights models/weights/gesture_classifier.pt \
                                --data data/processed/gestures \
                                --device cpu
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/weights/gesture_classifier.pt",
                   help="Path to YOLOv8-cls .pt weights file")
    p.add_argument("--data", default="data/gesture/dataset",
                   help="Dataset root (expects test/ sub-folder with one dir per class)")
    p.add_argument("--imgsz", type=int, default=224,
                   help="Input resolution (must match training)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu",
                   help="'cpu', '0', '0,1', etc.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    print(f"Running validation on: {args.data}")
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="test",           # use the test/ sub-folder
        plots=True,             # saves confusion matrix + other plots to runs/
        verbose=True,
    )

    # model.val() prints a full report; surface the headline numbers here too.
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"  Top-1 accuracy : {metrics.top1:.4f}")
    print(f"  Top-5 accuracy : {metrics.top5:.4f}")
    print("Confusion matrix and per-class plots saved under runs/classify/val*/")


if __name__ == "__main__":
    main()
