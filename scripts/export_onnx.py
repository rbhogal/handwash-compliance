"""
Export both models to ONNX for SiMa Palette SDK ingestion.
Both models are YOLOv8, so both use ultralytics' built-in export().

Usage:
    python scripts/export_onnx.py --detection models/weights/yolov8_detection.pt
    python scripts/export_onnx.py --gesture   models/weights/gesture_classifier.pt
    python scripts/export_onnx.py --all
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def export_yolo(weights: str, out_dir: Path, imgsz: int, label: str) -> None:
    """Generic YOLOv8 ONNX export (works for detect and classify tasks)."""
    from ultralytics import YOLO

    model = YOLO(weights)
    # ultralytics saves the .onnx next to the .pt file
    model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)

    onnx_path = Path(weights).with_suffix(".onnx")
    dest = out_dir / onnx_path.name
    shutil.move(str(onnx_path), dest)
    print(f"{label} ONNX → {dest}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 models to ONNX")
    p.add_argument("--detection", default=None, help="YOLOv8 detection .pt weights")
    p.add_argument("--gesture",   default=None, help="YOLOv8-classify gesture .pt weights")
    p.add_argument("--all",       action="store_true", help="Export both models")
    p.add_argument("--out_dir",   default="models/exports")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all or args.detection:
        w = args.detection or "models/weights/yolov8_detection.pt"
        export_yolo(w, out_dir, imgsz=640, label="Detection")

    if args.all or args.gesture:
        w = args.gesture or "models/weights/gesture_classifier.pt"
        export_yolo(w, out_dir, imgsz=224, label="Gesture classifier")
