"""
Dataset preparation for YOLOv8-classify training.

YOLOv8 classify expects this flat layout on disk:
    data/processed/gestures/
        train/
            palm_to_palm/
                img001.jpg
                img002.jpg
                ...
            right_over_left/
                ...
            ...   (one folder per WHO step, 0–6)
        val/
            ...
        test/
            ...

Run the helper below to convert raw clip folders into this format:
    python -c "from handwash.gesture.dataset import prepare_yolo_dataset; \
               prepare_yolo_dataset('data/raw/gestures', 'data/processed/gestures')"

Raw clip layout assumed:
    data/raw/gestures/
        0_palm_to_palm/
            clip_001/
                frame_000.jpg  ...
        1_right_over_left/
            ...
"""

from __future__ import annotations
import random
import shutil
from pathlib import Path
from typing import List, Tuple


STEP_NAMES = [
    "palm_to_palm",
    "right_over_left",
    "left_over_right",
    "interlaced",
    "thumbs",
    "fingertips_to_palm",
    "wrist",
]


def prepare_yolo_dataset(
    raw_root: str,
    out_root: str,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> None:
    """
    Convert raw clip folders into the flat train/val/test structure
    that YOLOv8 classify expects.

    Args:
        raw_root:  Path containing class subfolders with clip subdirs.
        out_root:  Destination root (will be created / overwritten).
        splits:    (train, val, test) fractions — must sum to 1.
        seed:      Random seed for reproducible splits.
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1"
    random.seed(seed)

    raw = Path(raw_root)
    out = Path(out_root)
    split_names = ("train", "val", "test")

    for class_dir in sorted(raw.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.split("_", 1)[-1]  # strip leading digit
        # Collect all frames across all clips in this class
        frames: List[Path] = []
        for clip_dir in sorted(class_dir.iterdir()):
            if clip_dir.is_dir():
                frames.extend(sorted(clip_dir.glob("*.jpg")))
                frames.extend(sorted(clip_dir.glob("*.png")))

        random.shuffle(frames)
        n = len(frames)
        n_train = int(n * splits[0])
        n_val   = int(n * splits[1])

        buckets = {
            "train": frames[:n_train],
            "val":   frames[n_train : n_train + n_val],
            "test":  frames[n_train + n_val :],
        }

        for split, split_frames in buckets.items():
            dest_dir = out / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_frames:
                shutil.copy2(src, dest_dir / src.name)

        print(f"  {class_name}: {n} frames → "
              f"train={len(buckets['train'])} "
              f"val={len(buckets['val'])} "
              f"test={len(buckets['test'])}")

    print(f"\nDataset written to {out}")
    _write_yaml(out)


def _write_yaml(out: Path) -> None:
    """Write the dataset YAML that YOLOv8 training reads."""
    yaml_lines = [
        f"path: {out.resolve()}",
        "train: train",
        "val:   val",
        "test:  test",
        "",
        f"nc: {len(STEP_NAMES)}",
        f"names: {STEP_NAMES}",
    ]
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines))
    print(f"Dataset YAML → {yaml_path}")
