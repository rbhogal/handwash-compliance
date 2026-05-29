# Handwashing Compliance Monitoring System

Real-time, two-layer hand hygiene compliance system for clinical environments.  
Deployed on **SiMa.ai Modalix** edge hardware with dual-camera input.

## Architecture

```
Wide-angle camera (1280×720)         Top-down camera (640×640)
        │                                      │
[1] YOLOv8 detection + ByteTrack         (no tracking)
    → persons, sinks                           │
        │                                      │
[2] Zone rule engine                           │
    → entry / sink / exit polygons             │
        │                                      │
        └──── person in sink zone? ───────────►│
                                        [3] Gesture classifier
                                            WHO 6-step scoring
                                               │
                        [4] Compliance verdict → LED + dashboard
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with two cameras (indices from config.yaml)
python scripts/run_pipeline.py

# 3. Override camera sources on the command line
python scripts/run_pipeline.py --source 0 --source-topdown 2

# 4. Test with video files (no cameras needed)
python scripts/run_pipeline.py \
    --source tests/data/wide.mp4 \
    --source-topdown tests/data/topdown.mp4
```

Dashboard auto-launches at **http://localhost:5000**.

## Training

**Detection (person + sink):**
```bash
# 1. Download + prepare dataset
python scripts/prepare_detection_data.py --max-samples 500

# 2. Train
python src/handwash/detection/train.py \
    --data config/detection_data.yaml \
    --weights yolov8n.pt \
    --epochs 50 --batch 8 --device mps --copy-weights
```

**Gesture Classifier (WHO 6 steps):**
```bash
# 1. Preprocess METC dataset
python scripts/prepare_gesture_data.py

# 2. Train
python src/handwash/gesture/train.py \
    --weights yolov8n-cls.pt \
    --epochs 50 --batch 32
```

## Datasets

| Layer | Dataset | Source |
|-------|---------|--------|
| Detection | COCO 2017 (person) + Open Images V7 (sink) | Downloaded via fiftyone |
| Gesture | METC subset — 212 videos, 6 WHO steps | https://zenodo.org/records/5808789 |

## Gesture Classes (6 WHO steps)

| Class | Label | WHO Step |
|-------|-------|----------|
| 0 | palm_to_palm | Step 2 — Rub hands palm to palm |
| 1 | palm_over_dorsum | Step 3 — Right palm over left dorsum |
| 2 | fingers_interlaced | Step 4 — Palm to palm fingers interlaced |
| 3 | backs_of_fingers | Step 5 — Backs of fingers to opposing palms |
| 4 | rotational_thumb | Step 6 — Rotational rubbing of thumb |
| 5 | fingertips_to_palm | Step 7 — Rotational rubbing of fingertips |

## ONNX Export (for Modalix)

```bash
python scripts/export_onnx.py --all
```

## Project Structure

```
src/handwash/
├── detection/        # Layer 1 — YOLOv8 detector + training
├── tracking/         # Layer 1 — ByteTrack wrapper
├── gesture/          # Layer 2 — YOLOv8-classify gesture classifier + training
├── compliance/       # Zone engine, state machine, LED, dashboard
└── pipeline.py       # Dual-camera pipeline orchestrator

scripts/
├── run_pipeline.py             # Run full system
├── prepare_detection_data.py   # Download + prepare detection dataset
├── prepare_gesture_data.py     # Preprocess METC gesture dataset
├── export_onnx.py              # Export to ONNX for Modalix
└── evaluate.py                 # Benchmark + evaluate models

config/
├── config.yaml           # Cameras, zones, thresholds, model paths
├── detection_data.yaml   # Detection training dataset config
└── gesture_data.yaml     # Gesture training dataset config
```

## Tests

```bash
pytest tests/
```

## Team

| Person | Module | Key File |
|--------|--------|----------|
| 1 | Detection | `src/handwash/detection/` |
| 2 | Tracking | `src/handwash/tracking/` |
| 3 | Gesture Classifier | `src/handwash/gesture/` |
| 4 | Compliance + Output | `src/handwash/compliance/` |
