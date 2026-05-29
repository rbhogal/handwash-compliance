"""
Preprocess METC gesture dataset for YOLOv8-classify training.

Input structure (data/gesture/raw/metc/):
    Interface_number_{1,2,3}/
        Videos/          ← .mp4 files
        Annotations/
            Annotator_1/
                <video_name>.json   ← frame-level WHO codes

Annotation codes (from METC):
    0 = other/transition  → discarded
    1 = palm_to_palm      → class 0
    2 = palm_over_dorsum  → class 1
    3 = fingers_interlaced → class 2
    4 = backs_of_fingers  → class 3
    5 = rotational_thumb  → class 4
    6 = fingertips_to_palm → class 5

Output structure (data/gesture/dataset/):
    train/
        palm_to_palm/        *.jpg
        palm_over_dorsum/    *.jpg
        fingers_interlaced/  *.jpg
        backs_of_fingers/    *.jpg
        rotational_thumb/    *.jpg
        fingertips_to_palm/  *.jpg
    val/   (same structure)
    test/  (same structure)

Usage:
    python scripts/prepare_gesture_data.py
    python scripts/prepare_gesture_data.py --frame-step 3 --val-split 0.15 --test-split 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
METC_DIR = ROOT / "data" / "gesture" / "raw" / "metc"
OUT_DIR = ROOT / "data" / "gesture" / "dataset"

# METC annotation code → (class_id, class_name)
# Code 0 = "other" — discarded
CODE_TO_CLASS = {
    1: (0, "palm_to_palm"),
    2: (1, "palm_over_dorsum"),
    3: (2, "fingers_interlaced"),
    4: (3, "backs_of_fingers"),
    5: (4, "rotational_thumb"),
    6: (5, "fingertips_to_palm"),
}

CLASS_NAMES = [name for _, name in sorted(CODE_TO_CLASS.values())]
INTERFACES = ["Interface_number_1", "Interface_number_2", "Interface_number_3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare METC gesture dataset for YOLOv8-classify")
    p.add_argument(
        "--metc-dir",
        default=str(METC_DIR),
        help="Path to METC raw data (default: data/gesture/raw/metc)",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help="Output dataset directory (default: data/gesture/dataset)",
    )
    p.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Extract every Nth frame (default: 3). Lower = more frames but more near-duplicates.",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction for validation set (default: 0.15)",
    )
    p.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction for test set (default: 0.15)",
    )
    return p.parse_args()


def load_annotation(json_path: Path) -> list[dict]:
    """Load frame-level annotation codes from a METC JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    return data["labels"]


def extract_frames(
    video_path: Path,
    labels: list[dict],
    frame_step: int,
) -> list[tuple[int, int]]:
    """
    Return list of (frame_index, class_id) for frames worth extracting.
    Skips code=0 (other) frames and samples every frame_step frames.
    """
    entries = []
    for i, label in enumerate(labels):
        if i % frame_step != 0:
            continue
        code = label.get("code", 0)
        if code not in CODE_TO_CLASS:
            continue
        class_id = CODE_TO_CLASS[code][0]
        entries.append((i, class_id))
    return entries


def process_video(
    video_path: Path,
    json_path: Path,
    frame_step: int,
    out_root: Path,
    split: str,
) -> int:
    """Extract labelled frames from one video into out_root/split/class_name/."""
    if not json_path.exists():
        return 0

    labels = load_annotation(json_path)
    entries = extract_frames(video_path, labels, frame_step)
    if not entries:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    saved = 0
    frame_idx = 0
    entry_map = {fi: ci for fi, ci in entries}
    max_frame = max(entry_map.keys())

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in entry_map:
            class_id = entry_map[frame_idx]
            class_name = CLASS_NAMES[class_id]
            out_dir = out_root / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{video_path.stem}_f{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_dir / fname), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def collect_video_paths(metc_dir: Path) -> list[tuple[Path, Path]]:
    """
    Walk all 3 interfaces and return (video_path, json_path) pairs.
    Matches each .mp4 to its annotation JSON by filename stem.
    """
    pairs = []
    for interface in INTERFACES:
        video_dir = metc_dir / interface / "Videos"
        ann_dir = metc_dir / interface / "Annotations" / "Annotator_1"

        if not video_dir.exists():
            print(f"  [WARN] Missing: {video_dir}")
            continue

        for video_path in sorted(video_dir.glob("*.mp4")):
            json_path = ann_dir / (video_path.stem + ".json")
            pairs.append((video_path, json_path))

    return pairs


def split_pairs(
    pairs: list,
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> dict[str, list]:
    """Randomly split video pairs into train/val/test."""
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test:n_test + n_val],
        "train": shuffled[n_test + n_val:],
    }


def write_dataset_yaml(out_dir: Path) -> None:
    yaml_content = f"""\
# Gesture classifier dataset — METC subset
# Auto-generated by scripts/prepare_gesture_data.py
# Classes align to METC annotation codes 1-6 (code 0 discarded)

path: {out_dir.resolve()}
train: train
val:   val
test:  test

nc: 6
names:
  0: palm_to_palm
  1: palm_over_dorsum
  2: fingers_interlaced
  3: backs_of_fingers
  4: rotational_thumb
  5: fingertips_to_palm
"""
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    config_yaml = ROOT / "config" / "gesture_data.yaml"
    config_yaml.write_text(yaml_content)

    print(f"\nDataset YAML written to:\n  {yaml_path}\n  {config_yaml}")


def main() -> None:
    args = parse_args()
    metc_dir = Path(args.metc_dir)
    out_dir = Path(args.output_dir)

    print(f"METC source : {metc_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Frame step  : every {args.frame_step} frames")
    print(f"Split       : {1 - args.val_split - args.test_split:.0%} train / "
          f"{args.val_split:.0%} val / {args.test_split:.0%} test")
    print()

    pairs = collect_video_paths(metc_dir)
    print(f"Found {len(pairs)} videos across {len(INTERFACES)} interfaces")

    splits = split_pairs(pairs, args.val_split, args.test_split)
    print(f"Split: {len(splits['train'])} train / {len(splits['val'])} val / "
          f"{len(splits['test'])} test videos\n")

    total_frames = 0
    for split, split_pairs_list in splits.items():
        split_frames = 0
        for video_path, json_path in tqdm(split_pairs_list, desc=f"Processing {split}"):
            saved = process_video(video_path, json_path, args.frame_step, out_dir, split)
            split_frames += saved
        print(f"  {split}: {split_frames} frames saved")
        total_frames += split_frames

    print(f"\nTotal frames extracted: {total_frames}")

    write_dataset_yaml(out_dir)

    print("\n✅ Done. To train:\n")
    print(
        "  python src/handwash/gesture/train.py \\\n"
        "    --data config/gesture_data.yaml \\\n"
        "    --weights yolov8n-cls.pt \\\n"
        "    --epochs 50 \\\n"
        "    --batch 32 \\\n"
        "    --device mps\n"
    )


if __name__ == "__main__":
    main()
