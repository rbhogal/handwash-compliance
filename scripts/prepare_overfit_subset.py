"""
Create a 10% subset of the gesture training data for intentional overfitting.

Purpose: verify the model CAN learn from this data before committing to a
full PSKUS training run. If the model overfits the subset (train accuracy
90%+), the pipeline and data are confirmed working correctly.

Usage:
    # 1. Create the subset
    python scripts/prepare_overfit_subset.py

    # 2. Train on it (no dropout, more epochs to force overfitting)
    python src/handwash/gesture/train.py \
        --data data/gesture/overfit_subset \
        --epochs 100 \
        --batch 16 \
        --name overfit_test

Expected result: training accuracy should reach 90%+ within 20-30 epochs.
If it doesn't, something is wrong with the data or model architecture.
"""

import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "gesture" / "dataset" / "train"
OUT_DIR = ROOT / "data" / "gesture" / "overfit_subset"
FRACTION = 0.10


def main() -> None:
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR} not found. Run prepare_gesture_data.py first.")
        return

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    class_dirs = sorted([d for d in SRC_DIR.iterdir() if d.is_dir()])
    print(f"Building {FRACTION:.0%} overfit subset from {SRC_DIR.name}/\n")

    total = 0
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.jpg"))
        n = max(1, int(len(images) * FRACTION))
        sampled = random.sample(images, n)

        # train/ — the 10% subset we want to overfit on
        out_train = OUT_DIR / "train" / class_dir.name
        out_train.mkdir(parents=True, exist_ok=True)
        for img in sampled:
            shutil.copy2(img, out_train / img.name)

        # val/ — ultralytics requires it; reuse a few train images (fine for overfit test)
        out_val = OUT_DIR / "val" / class_dir.name
        out_val.mkdir(parents=True, exist_ok=True)
        for img in sampled[:5]:
            shutil.copy2(img, out_val / img.name)

        print(f"  {class_dir.name}: {n} train / 5 val")
        total += n

    print(f"\nTotal: {total} images → {OUT_DIR}")
    print("\nTo train (intentional overfit — no dropout, more epochs):")
    print(
        "\n  python src/handwash/gesture/train.py \\\n"
        "    --data data/gesture/overfit_subset \\\n"
        "    --epochs 100 \\\n"
        "    --batch 16 \\\n"
        "    --name overfit_test\n"
    )
    print("Expected: train accuracy 90%+ within 20-30 epochs.")
    print("If not reached → investigate data pipeline before running full PSKUS training.")


if __name__ == "__main__":
    main()
