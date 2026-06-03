"""
Randomly sample images from the gesture dataset and save them with
class labels drawn on — for manual visual inspection of annotation quality.

Usage:
    python scripts/inspect_gesture_dataset.py
    python scripts/inspect_gesture_dataset.py --n 150 --split train

Output:
    data/gesture/inspection/   ← open this folder and visually check images
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "gesture" / "dataset"
OUT_DIR = ROOT / "data" / "gesture" / "inspection"

# One colour per class for easy visual distinction
CLASS_COLOURS = {
    "palm_to_palm":       (0,   255, 0),    # green
    "palm_over_dorsum":   (255, 128, 0),    # orange
    "fingers_interlaced": (0,   0,   255),  # blue
    "backs_of_fingers":   (255, 0,   255),  # magenta
    "rotational_thumb":   (0,   255, 255),  # cyan
    "fingertips_to_palm": (255, 255, 0),    # yellow
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect gesture dataset annotations visually")
    p.add_argument("--n",     type=int, default=150, help="Total images to sample (default: 150)")
    p.add_argument("--split", default="train",       help="Dataset split to sample from (default: train)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = DATASET_DIR / args.split

    if not split_dir.exists():
        print(f"ERROR: {split_dir} does not exist. Run prepare_gesture_data.py first.")
        return

    # Collect all images per class
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"ERROR: No class folders found in {split_dir}")
        return

    # Sample evenly across classes
    per_class = max(1, args.n // len(class_dirs))
    print(f"Sampling ~{per_class} images per class from {args.split}/ ({len(class_dirs)} classes)")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    total_saved = 0
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg"))

        if not images:
            print(f"  [WARN] No images in {class_name}")
            continue

        sampled = random.sample(images, min(per_class, len(images)))
        colour = CLASS_COLOURS.get(class_name, (255, 255, 255))

        for img_path in sampled:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Draw filename small at top for traceability
            cv2.putText(img, img_path.name, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # Draw class label bar at bottom
            cv2.rectangle(img, (0, h - 36), (w, h), colour, -1)
            cv2.putText(img, class_name, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            out_name = f"{class_name}__{img_path.name}"
            cv2.imwrite(str(OUT_DIR / out_name), img)
            total_saved += 1

    print(f"\nSaved {total_saved} images → {OUT_DIR}")
    print("Open that folder and visually verify:")
    print("  ✓ Label colour matches what's actually happening in the image")
    print("  ✓ Hands are visible and in frame")
    print("  ✓ No obviously wrong labels")
    print(f"\n  open {OUT_DIR}")


if __name__ == "__main__":
    main()
