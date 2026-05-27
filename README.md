# Handwashing Compliance Monitoring System

Real-time, two-layer hand hygiene compliance system for clinical environments.  
Deployed on **SiMa.ai Modalix** edge hardware.

## Architecture

```
Camera feed
    ↓
[1] YOLOv8 detection    → persons, sinks, dispensers
    ↓
[2] ByteTrack           → persistent anonymised IDs + trajectories
    ↓
[3] Zone rule engine    → is this person at the sink?
    ↓ (if yes)
[4] Gesture classifier  → WHO 7-step quality scoring (MobileNetV3 + LSTM)
    ↓
[5] Compliance verdict  → LED output + logs + dashboard
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline (webcam)
python scripts/run_pipeline.py

# 3. Run on a recorded clip
python scripts/run_pipeline.py --source path/to/video.mp4
```

Dashboard auto-launches at **http://localhost:5000**.

## Project Structure

```
src/handwash/
├── detection/        # Layer 1 — YOLOv8 detector + training
├── tracking/         # Layer 1 — ByteTrack wrapper
├── gesture/          # Layer 2 — MobileNetV3+LSTM classifier + dataset + training
├── compliance/       # Rule engine, state machine, LED, dashboard
└── pipeline.py       # Main orchestrator

scripts/
├── run_pipeline.py   # Run full system
├── export_onnx.py    # Export to ONNX for Modalix
└── evaluate.py       # Evaluate gesture classifier

config/
└── config.yaml       # Zones, thresholds, camera, model paths
```

## Training

**Detection:**
```bash
python src/handwash/detection/train.py --data config/detection_data.yaml
```

**Gesture Classifier:**
```bash
python src/handwash/gesture/train.py --data data/processed/gestures --epochs 30
```

## ONNX Export (for Modalix)

```bash
python scripts/export_onnx.py --all
```

## Tests

```bash
pytest tests/
```

## Team

| Person | Module | Key File |
|---|---|---|
| 1 | Detection | `src/handwash/detection/` |
| 2 | Tracking | `src/handwash/tracking/` |
| 3 | Gesture Classifier | `src/handwash/gesture/` |
| 4 | Compliance + Output | `src/handwash/compliance/` |
