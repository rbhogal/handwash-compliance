"""
Layer 1 — Multi-Object Tracking
ByteTrack wrapper that maintains persistent anonymized IDs across frames.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker   # bundled in ultralytics


@dataclass
class Track:
    """Represents one tracked person across frames."""
    track_id: int
    bbox: np.ndarray           # [x1, y1, x2, y2]
    centroid: np.ndarray       # [cx, cy]
    confidence: float
    age: int = 0               # frames since first seen
    history: List[np.ndarray] = field(default_factory=list)  # centroid history

    def update(self, bbox: np.ndarray, confidence: float) -> None:
        self.bbox = bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.centroid = np.array([cx, cy])
        self.confidence = confidence
        self.age += 1
        self.history.append(self.centroid.copy())


class PersonTracker:
    """
    Wraps YOLOv8's built-in ByteTrack to produce a dict of active Track objects.

    Usage:
        tracker = PersonTracker(weights="models/weights/yolov8_detection.pt")
        tracks = tracker.update(frame)   # {track_id: Track}
    """

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        tracker_config: str = "bytetrack.yaml",
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(weights)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config

        self._active: Dict[int, Track] = {}

    def update(self, frame: np.ndarray) -> Dict[int, Track]:
        """
        Run detection + ByteTrack on one frame.

        Returns:
            Dict mapping track_id → Track for all currently active persons.
        """
        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_config,
            persist=True,
            classes=[0],       # persons only
            verbose=False,
        )

        seen_ids: set[int] = set()

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for box, tid in zip(result.boxes, result.boxes.id):
                track_id = int(tid.item())
                bbox = box.xyxy.squeeze().cpu().numpy()
                conf = float(box.conf.item())

                if track_id not in self._active:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    self._active[track_id] = Track(
                        track_id=track_id,
                        bbox=bbox,
                        centroid=np.array([cx, cy]),
                        confidence=conf,
                    )
                else:
                    self._active[track_id].update(bbox, conf)

                seen_ids.add(track_id)

        # Prune tracks no longer visible
        lost = [tid for tid in self._active if tid not in seen_ids]
        for tid in lost:
            del self._active[tid]

        return dict(self._active)

    def reset(self) -> None:
        """Clear all active tracks (e.g. between video clips)."""
        self._active.clear()
