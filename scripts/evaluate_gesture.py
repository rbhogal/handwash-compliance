"""
Evaluate gesture classifier on the METC test set (per-class accuracy).

Usage:
    python scripts/evaluate_gesture.py
    python scripts/evaluate_gesture.py --data data/gesture/dataset/test
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from handwash.gesture.classifier import GestureClassifier, WHO_STEP_LABELS

LABEL_TO_ID = {v: k for k, v in WHO_STEP_LABELS.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gesture/dataset/test")
    parser.add_argument("--weights", default="models/weights/gesture_classifier.pt")
    parser.add_argument("--conf", type=float, default=0.0)  # 0 = accept all predictions
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists():
        sys.exit(f"Test set not found: {data_dir}")

    # smoothing_window=1 — single-frame evaluation, no temporal context
    clf = GestureClassifier(weights=args.weights, smoothing_window=1, conf_threshold=args.conf)

    correct = defaultdict(int)
    total = defaultdict(int)
    skipped = 0

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        label = class_dir.name
        if label not in LABEL_TO_ID:
            continue
        true_id = LABEL_TO_ID[label]
        images = sorted(class_dir.glob("*.jpg"))
        print(f"Evaluating {label} ({len(images)} frames)...", flush=True)
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                skipped += 1
                continue
            clf.push_frame(frame)
            pred = clf.classify()
            total[label] += 1
            if pred is not None and pred.step_id == true_id:
                correct[label] += 1

    print("\n── Per-class accuracy ──────────────────────────")
    overall_correct = 0
    overall_total = 0
    for i in range(6):
        label = WHO_STEP_LABELS[i]
        t = total[label]
        c = correct[label]
        acc = (c / t * 100) if t > 0 else 0.0
        overall_correct += c
        overall_total += t
        print(f"  Step {i+1}/6 {label:<25} {c:>5}/{t:<5} {acc:5.1f}%")

    overall = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0
    print(f"\n  Overall Top-1 accuracy: {overall_correct}/{overall_total} = {overall:.1f}%")
    if skipped:
        print(f"  Skipped (unreadable): {skipped}")


if __name__ == "__main__":
    main()
