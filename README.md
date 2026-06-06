# Handwashing Compliance Monitoring System

Real-time, two-layer hand hygiene compliance system for clinical environments.  
Deployed on **SiMa.ai Modalix** edge hardware with dual-camera input.

## Architecture

```
Wide-angle camera (1920×1080)        Top-down camera (640×640)
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

## Gesture Classifier Results

The gesture classifier was fine-tuned from a PSKUS-trained baseline onto the METC dataset using Colab (`notebooks/finetune_metc.ipynb`).

| Model                                 | Test Accuracy |
| ------------------------------------- | ------------- |
| Trained on PSKUS → fine-tuned on METC | **71.4%**     |
| Original paper (Xception, METC)       | 66.8%         |

Per-class accuracy on METC test set (8,054 frames):

| Step | Label              | Accuracy |
| ---- | ------------------ | -------- |
| 1    | palm_to_palm       | 74.6%    |
| 2    | palm_over_dorsum   | 67.3%    |
| 3    | fingers_interlaced | 53.3%    |
| 4    | backs_of_fingers   | 77.3%    |
| 5    | rotational_thumb   | 73.0%    |
| 6    | fingertips_to_palm | 81.2%    |

Reference: [Ivanovs et al. 2020 — Automated Quality Assessment of Hand Washing Using Deep Learning](https://arxiv.org/abs/2011.11383)

## Training

**Detection (person + sink):**

```bash
python scripts/prepare_detection_data.py --max-samples 500
python src/handwash/detection/train.py \
    --data config/detection_data.yaml \
    --weights yolov8n.pt \
    --epochs 50 --batch 8 --device mps --copy-weights
```

**Gesture Classifier (WHO 6 steps):**

Training was done in two stages on Google Colab using `notebooks/finetune_metc.ipynb`:

1. **Stage 1 — Train on PSKUS dataset** (3,185 videos, 6 WHO steps)
   - Base model: `yolov8n-cls.pt` (ImageNet pretrained)
   - 50 epochs, batch=32

2. **Stage 2 — Fine-tune on METC dataset** (212 videos, ~48k frames)
   - Starts from PSKUS weights
   - 50 epochs, lr=1e-4, dropout=0.2
   - Best weights saved as `models/weights/gesture_classifier.pt`

## Datasets

| Layer             | Dataset                                    | Source                             |
| ----------------- | ------------------------------------------ | ---------------------------------- |
| Detection         | COCO 2017 (person) + Open Images V7 (sink) | Downloaded via fiftyone            |
| Gesture (Stage 1) | PSKUS — 3,185 videos, 6 WHO steps          | https://zenodo.org/records/4537209 |
| Gesture (Stage 2) | METC subset — 212 videos, 6 WHO steps      | https://zenodo.org/records/5808789 |

## Gesture Classes (6 WHO steps)

| Class | Label              | WHO Step                                    |
| ----- | ------------------ | ------------------------------------------- |
| 0     | palm_to_palm       | Step 2 — Rub hands palm to palm             |
| 1     | palm_over_dorsum   | Step 3 — Right palm over left dorsum        |
| 2     | fingers_interlaced | Step 4 — Palm to palm fingers interlaced    |
| 3     | backs_of_fingers   | Step 5 — Backs of fingers to opposing palms |
| 4     | rotational_thumb   | Step 6 — Rotational rubbing of thumb        |
| 5     | fingertips_to_palm | Step 7 — Rotational rubbing of fingertips   |

## Zone Calibration

Zone polygons (entry, sink, exit) are defined in `config/config.yaml` as pixel coordinates for the wide-angle camera. To recalibrate for a new camera position:

```bash
python scripts/calibrate_zones.py --frame path/to/frame.jpg
```

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
├── calibrate_zones.py          # Interactive zone polygon calibration
├── test_gesture.py             # Test gesture classifier on a video
├── evaluate_gesture.py         # Evaluate per-class accuracy on test set
├── prepare_detection_data.py   # Download + prepare detection dataset
├── prepare_gesture_data.py     # Preprocess METC gesture dataset
└── export_onnx.py              # Export to ONNX for Modalix

notebooks/
└── finetune_metc.ipynb         # Colab fine-tuning notebook (PSKUS → METC)

config/
├── config.yaml           # Cameras, zones, thresholds, model paths
├── detection_data.yaml   # Detection training dataset config
└── gesture_data.yaml     # Gesture training dataset config
```

## Tests

```bash
pytest tests/
```
