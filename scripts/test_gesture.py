"""
Quick test: run gesture classifier on a video and report which WHO steps were detected.

Usage:
    python scripts/test_gesture.py
    python scripts/test_gesture.py --video path/to/topdown.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from handwash.gesture.classifier import GestureClassifier, WHO_STEP_LABELS

VIDEO = "tests/data/topdown/2021-07-01_12-30-23-31-1.mp4"
WEIGHTS = "models/weights/gesture_classifier.pt"


def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--weights", default=WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.6)
    args = parser.parse_args()

    clf = GestureClassifier(weights=args.weights, smoothing_window=8, conf_threshold=args.conf)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    steps_detected = set()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        clf.push_frame(frame)
        if clf.ready():
            pred = clf.classify()
            if pred is not None:
                steps_detected.add(pred.step_id)
        frame_count += 1

    cap.release()

    print(f"\nFrames processed: {frame_count}")
    print(f"Steps detected:   {len(steps_detected)}/6")
    print()
    for i in range(6):
        label = WHO_STEP_LABELS[i]
        status = "YES" if i in steps_detected else "NO"
        print(f"  Step {i+1}/6 {label}: {status}")

    print()
    if steps_detected == set(range(6)):
        print("All 6 steps detected.")
    else:
        missing = [WHO_STEP_LABELS[i] for i in range(6) if i not in steps_detected]
        print(f"Missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
