"""
Download and prepare the Layer 1 detection dataset.

Merges three data sources into one 3-class YOLO dataset:
    0 = person        (COCO 2017)
    1 = sink          (Open Images V7 — includes kitchen + clinical sinks)
    2 = soap_dispenser (Open Images V7)

Output layout (under --output-dir):
    detection_dataset/
        images/
            train/   *.jpg
            val/     *.jpg
        labels/
            train/   *.txt   (YOLO format)
            val/     *.txt
        dataset.yaml         (written here, copied to config/detection_data.yaml)

Requirements:
    pip install fiftyone opencv-python tqdm

Usage:
    # Quick test — 200 images per class
    python scripts/prepare_detection_data.py --max-samples 200

    # Full dataset
    python scripts/prepare_detection_data.py --max-samples 2000

    # Skip re-download if data already present
    python scripts/prepare_detection_data.py --max-samples 500 --skip-download
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare detection training data")
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "detection_dataset"),
        help="Root directory for the prepared dataset (default: data/detection_dataset)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max images to download per class (default: 500). Use 2000+ for a serious run.",
    )
    p.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation (default: 0.15)",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip fiftyone download; assume raw data already in --output-dir/raw/",
    )
    return p.parse_args()


# ── Class mapping ──────────────────────────────────────────────────────────────

# Our YOLO class IDs — 2 classes for MVP
# NOTE: soap_dispenser intentionally excluded.
# Open Images V7 has very few soap dispenser examples (~58) and Roboflow
# alternatives are unreliable (mix soaps and dispensers separately).
# The HandwashDetector is not called at runtime anyway — zones are polygon-based.
# Add soap_dispenser as class 2 in a future iteration with purpose-collected data.
CLASS_MAP = {
    "person": 0,
    "sink": 1,
}

# Open Images uses these exact label strings (case-sensitive)
OI_LABELS = {
    "sink": "Sink",
    # "soap_dispenser": "Soap dispenser",  # excluded — see CLASS_MAP note above
}

# COCO label string (fiftyone uses lowercase)
COCO_PERSON_LABEL = "person"


# ── Download helpers ───────────────────────────────────────────────────────────

def download_open_images(raw_dir: Path, max_samples: int) -> None:
    """Download Sink + Soap dispenser images from Open Images V7 via fiftyone."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    for our_name, oi_label in OI_LABELS.items():
        for split in ("train", "validation"):
            out_split = "val" if split == "validation" else "train"
            print(f"\n[Open Images] Downloading '{oi_label}' ({out_split}) ...")
            ds_name = f"oi_{our_name}_{out_split}"

            if fo.dataset_exists(ds_name):
                print(f"  Dataset '{ds_name}' already exists — skipping download.")
                ds = fo.load_dataset(ds_name)
            else:
                ds = foz.load_zoo_dataset(
                    "open-images-v7",
                    split=split,
                    label_types=["detections"],
                    classes=[oi_label],
                    max_samples=max_samples,
                    dataset_name=ds_name,
                )

            _export_oi_to_raw(ds, raw_dir / out_split / our_name, oi_label)
            ds.save()


def download_coco_person(raw_dir: Path, max_samples: int) -> None:
    """Download person images from COCO 2017 via fiftyone."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    for split in ("train", "validation"):
        out_split = "val" if split == "validation" else "train"
        print(f"\n[COCO] Downloading 'person' ({out_split}) ...")
        ds_name = f"coco_person_{out_split}"

        if fo.dataset_exists(ds_name):
            print(f"  Dataset '{ds_name}' already exists — skipping download.")
            ds = fo.load_dataset(ds_name)
        else:
            ds = foz.load_zoo_dataset(
                "coco-2017",
                split=split,
                label_types=["detections"],
                classes=["person"],
                max_samples=max_samples,
                dataset_name=ds_name,
            )

        _export_coco_to_raw(ds, raw_dir / out_split / "person")
        ds.save()


# ── Export helpers ─────────────────────────────────────────────────────────────

def _export_oi_to_raw(ds, raw_out: Path, oi_label: str) -> None:
    """Save Open Images detections as (image_path, [(x1,y1,x2,y2)] in relative coords)."""
    import fiftyone as fo
    raw_out.mkdir(parents=True, exist_ok=True)
    ann_file = raw_out / "annotations.txt"

    with open(ann_file, "w") as f:
        for sample in ds:
            img_path = sample.filepath
            boxes = []
            if sample.ground_truth:
                for det in sample.ground_truth.detections:
                    if det.label == oi_label:
                        # fiftyone bounding_box: [x, y, w, h] already relative (0-1)
                        bx, by, bw, bh = det.bounding_box
                        boxes.append(f"{bx:.6f},{by:.6f},{bw:.6f},{bh:.6f}")
            if boxes:
                f.write(f"{img_path}|{'|'.join(boxes)}\n")

    print(f"  Saved {_count_lines(ann_file)} annotated images → {ann_file}")


def _export_coco_to_raw(ds, raw_out: Path) -> None:
    """Save COCO person detections as (image_path, [(x1,y1,x2,y2)] relative)."""
    raw_out.mkdir(parents=True, exist_ok=True)
    ann_file = raw_out / "annotations.txt"

    with open(ann_file, "w") as f:
        for sample in ds:
            img_path = sample.filepath
            boxes = []
            if sample.ground_truth:
                for det in sample.ground_truth.detections:
                    if det.label == "person":
                        bx, by, bw, bh = det.bounding_box
                        boxes.append(f"{bx:.6f},{by:.6f},{bw:.6f},{bh:.6f}")
            if boxes:
                f.write(f"{img_path}|{'|'.join(boxes)}\n")

    print(f"  Saved {_count_lines(ann_file)} annotated images → {ann_file}")


# ── Convert to YOLO format ─────────────────────────────────────────────────────

def build_yolo_dataset(raw_dir: Path, out_dir: Path, val_split: float) -> None:
    """
    Read raw annotation files and write YOLO-format labels + copy images.
    """
    import random
    from tqdm import tqdm

    random.seed(42)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_entries: list[tuple[str, int, list]] = []  # (img_path, class_id, boxes_relxywh)

    for split_raw in ("train", "val"):
        for class_name, class_id in CLASS_MAP.items():
            ann_file = raw_dir / split_raw / class_name / "annotations.txt"
            if not ann_file.exists():
                print(f"  [WARN] Missing annotation file: {ann_file}")
                continue
            with open(ann_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("|")
                    img_path = parts[0]
                    boxes = []
                    for b in parts[1:]:
                        vals = [float(v) for v in b.split(",")]
                        # fiftyone: [x, y, w, h] top-left relative → YOLO cx, cy, w, h
                        bx, by, bw, bh = vals
                        cx = bx + bw / 2
                        cy = by + bh / 2
                        boxes.append((cx, cy, bw, bh))
                    if boxes:
                        all_entries.append((img_path, class_id, boxes))

    random.shuffle(all_entries)
    n_val = int(len(all_entries) * val_split)
    splits = {"val": all_entries[:n_val], "train": all_entries[n_val:]}

    print(f"\nBuilding YOLO dataset: {len(splits['train'])} train, {len(splits['val'])} val images")

    for split, entries in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split

        for img_path_str, class_id, boxes in tqdm(entries, desc=f"Writing {split}"):
            img_path = Path(img_path_str)
            if not img_path.exists():
                continue

            stem = img_path.stem + "_" + str(class_id)
            dst_img = img_dir / (stem + img_path.suffix)
            dst_lbl = lbl_dir / (stem + ".txt")

            shutil.copy2(img_path, dst_img)

            with open(dst_lbl, "w") as f:
                for cx, cy, bw, bh in boxes:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Dataset written to {out_dir}")


# ── Write dataset YAML ─────────────────────────────────────────────────────────

def write_dataset_yaml(out_dir: Path) -> None:
    """Write the dataset.yaml consumed by ultralytics training."""
    yaml_content = f"""\
# Detection dataset — person + sink + soap_dispenser
# Auto-generated by scripts/prepare_detection_data.py

path: {out_dir.resolve()}
train: images/train
val:   images/val

nc: 2
names:
  0: person
  1: sink
# soap_dispenser (class 2) excluded from MVP — see CLASS_MAP in this script for rationale
"""
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)

    # Also copy to config/ so train.py default path works
    config_yaml = ROOT / "config" / "detection_data.yaml"
    config_yaml.write_text(yaml_content)

    print(f"\nDataset YAML written to:\n  {yaml_path}\n  {config_yaml}")


# ── Utils ──────────────────────────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw"

    print(f"Output directory : {out_dir}")
    print(f"Max samples/class: {args.max_samples}")
    print(f"Val split        : {args.val_split:.0%}")

    if not args.skip_download:
        try:
            import fiftyone  # noqa: F401
        except ImportError:
            print("\nERROR: fiftyone not installed. Run:\n  pip install fiftyone\n")
            sys.exit(1)

        download_open_images(raw_dir, args.max_samples)
        download_coco_person(raw_dir, args.max_samples)
    else:
        print("\n--skip-download set — skipping fiftyone download.")

    build_yolo_dataset(raw_dir, out_dir, args.val_split)
    write_dataset_yaml(out_dir)

    print("\n✅ Done. To train:\n")
    print(
        "  python src/handwash/detection/train.py \\\n"
        "    --data config/detection_data.yaml \\\n"
        "    --weights yolov8n.pt \\\n"
        "    --epochs 50 \\\n"
        "    --batch 16 \\\n"
        "    --device 0\n"
    )


if __name__ == "__main__":
    main()
