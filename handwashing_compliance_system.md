# Handwashing Compliance Monitoring System

## Objective
Develop a real-time, two-layer system that monitors hand hygiene compliance in clinical environments. The system combines zone-based compliance detection (did the person wash?) with WHO 7-step gesture quality scoring (did they wash *correctly?*), deployed on SiMa.ai Modalix edge hardware with a simple LED output for immediate feedback.

---

## Camera Setup (Dual-Camera)

The system uses two cameras, both supported natively by the Modalix chip's 4× MIPI CSI-2 inputs:

| Camera | Position | Role |
|--------|----------|------|
| **Wide-angle** | Mounted to cover entry + sink + exit area | Zone tracking — detects when a person enters, reaches the sink, and exits |
| **Top-down** | Mounted directly above the sink | Gesture classification — close-up of hands fed to the WHO 7-step classifier |

**Why two cameras:**
- The wide-angle view is too distorted for reliable gesture recognition
- The top-down view has no zone context (can't tell if person entered or exited)
- Each camera does one job well — zones or gestures, not both

---

## Models

**Layer 1 — Compliance Detection**
- Object Detection: YOLOv8 (person, sink, soap dispenser)
- Tracking: ByteTrack (wide-angle camera only)

**Layer 2 — Gesture Quality Scoring**
- Gesture Classifier: YOLOv8-classify
- Input: top-down camera frame (full frame or configured ROI crop)
- Activates only when a person is confirmed in the sink zone by the wide-angle tracker

---

## Datasets

**Layer 1 — Detection**
- COCO 2017 (person detection)
- Open Images V7 (sink — class 460, soap dispenser — class 475)
- Prepared via: `python scripts/prepare_detection_data.py`

**Layer 2 — Gesture Classifier**
- WHO Hand Wash Dataset — Kaggle (7-step gesture clips)
- PSKUS / Zenodo — 3,185 real hospital wash episodes, 30fps, 640×480, frame-level WHO annotations (18.4 GB)
- METC / Zenodo — 212 lab-controlled videos from 72 medical students, ~16fps, 640×480 (2.1 GB)
- All datasets require preprocessing: frame extraction → class folders → train/val/test split
- Prepared via: `python scripts/prepare_gesture_data.py` *(to be built)*

**Demo / Benchmarking footage**
- Top-down camera: PSKUS/METC video clips (same data used for training)
- Wide-angle camera: self-recorded clips (~5 clips, 30–60 sec each, any sink)

---

## Pipeline

```
Wide-angle camera (1280×720)         Top-down camera (640×640)
        │                                      │
[1] YOLOv8 detection                     (no detection)
        │                                      │
[2] ByteTrack → person IDs + trajectories      │
        │                                      │
[3] Zone rule engine                           │
    (entry / sink / exit polygons)             │
        │                                      │
        └──── person in sink zone? ───────────►│
                                        [4] Gesture classifier
                                            WHO step quality scoring
                                               │
                        ┌──────────────────────┘
                        │
                [5] Compliance verdict
                    LED output (green / red) + logs
                        │
              Side-by-side display
         [wide + zone overlays | top-down + gesture label]
```

**Detailed stages:**

1. **Input:** Two camera streams — wide-angle (zones) + top-down (gestures)
2. **Detection:** Detect persons on wide-angle frame; sink/dispenser detected for setup only
3. **Tracking:** ByteTrack assigns persistent anonymized IDs on wide-angle stream
4. **Zone Rule Engine:**
   - Entry detection (person enters zone)
   - Sink interaction detection (person centroid inside sink polygon)
   - Dwell time measurement (minimum 20 seconds)
   - Exit detection
5. **Gesture Quality Scoring (Layer 2):**
   - Triggered when wide-angle tracker confirms person is in sink zone
   - Top-down frame (or configured ROI crop) sent to gesture classifier
   - Classifies each of the 7 WHO steps in real-time
   - Temporal smoothing: majority vote over 8-frame window
6. **Compliance Verdict:**
   - Full compliance: all steps detected + dwell time met → green LED
   - Partial compliance: some steps missed or too brief → red LED + specific feedback
   - Non-compliance: no sink interaction → red LED
7. **Output:**
   - Side-by-side display: wide-angle with zone overlays (left) + top-down with gesture label (right)
   - LED indicator (immediate physical feedback at point of care)
   - Anonymized compliance logs (per wash event, no PII)
   - Aggregate compliance rate dashboard

---

## Rule-Set Design (Tracking + Temporal Logic)

**Zone Definition** (polygon-based, defined once per camera installation in `config/config.yaml`):
- Entry zone
- Sink zone
- Exit zone

Zones are in wide-angle pixel coordinates. Top-down camera has no zone definitions — it is gesture-only.

**State Machine per Track ID:**

```
Entered → At Sink → Washing → Exited
                            ↘ Violation (no sink interaction)
```

**Transitions:**
- Entered → At Sink: person centroid enters sink zone polygon (wide-angle)
- At Sink → Washing: dwell time exceeds threshold, gesture classifier activates (top-down)
- Washing → Exited: person leaves sink zone
- Entered → Exited (skipping sink): violation flagged

**Temporal Constraints:**
- Max allowed time to reach sink after entry
- Minimum required wash duration (20 seconds per WHO guidelines)
- Per-step minimum duration thresholds

---

## Team Structure (4 People)

| Person | Responsibility |
|--------|---------------|
| 1 | YOLOv8 detection — persons, sinks, dispensers (COCO + Open Images fine-tune) |
| 2 | ByteTrack integration — persistent IDs, trajectory tracking (wide-angle camera) |
| 3 | Gesture classifier — WHO 7-step quality scoring (PSKUS + METC + Kaggle datasets) |
| 4 | Rule engine + compliance logic + LED output + dashboard + dual-camera pipeline |

Person 3 develops and validates the gesture classifier independently on PSKUS/METC datasets before plugging into the live pipeline at integration stage.

---

## Hardware & Deployment

**Target:** SiMa.ai Modalix MLSoC (50 TOPS, 4× MIPI CSI-2 inputs, onboard CVU)

**Camera inputs used:** 2 of 4 available MIPI CSI-2 ports
- Port 1: Wide-angle camera (1280×720, 30fps)
- Port 2: Top-down camera (640×640, 30fps)

**Deployment path:** PyTorch → ONNX export → SiMa Palette SDK → Modalix on-chip inference

**Design constraints:**
- Standard PyTorch layers only (no custom ops) to ensure clean ONNX export
- ONNX export tested early, not left to the end
- MobileNetV3 backbone preferred over heavier architectures for real-time inference budget

---

## Running the Pipeline

```bash
# Use camera indices defined in config/config.yaml
python scripts/run_pipeline.py

# Override camera sources on the command line
python scripts/run_pipeline.py --source 0 --source-topdown 2

# Test with pre-recorded video files (no cameras needed)
python scripts/run_pipeline.py \
    --source tests/data/wide.mp4 \
    --source-topdown tests/data/topdown.mp4
```

---

## Benchmarks

- FPS and per-frame latency (target: 20–30 FPS on edge, dual-stream)
- Detection accuracy: mAP for person, sink, dispenser
- Tracking consistency: ID switch rate
- Gesture classification accuracy: per-class and overall (target: 90–95% on held-out PSKUS data)
- Compliance detection accuracy: precision/recall vs. ground truth wash events
- End-to-end event latency: time from wash completion to LED trigger

---

## Semester Timeline

| Month | Milestone |
|-------|-----------|
| 1 | Environment setup, datasets downloaded, baseline detection + tracking working on wide-angle, gesture classifier first training run on METC dataset |
| 2 | Fine-tuned detection model, stable dual-camera pipeline, gesture classifier with temporal smoothing, rule engine logic coded |
| 3 | Full dual-camera pipeline integration, end-to-end testing on recorded clips, ONNX export, first Modalix deployment |
| 4 | Benchmarking (dual-stream FPS/latency), demo video, live demonstration, documentation |
