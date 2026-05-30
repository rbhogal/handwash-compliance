"""
Layer 2 — Gesture Quality Scoring
YOLOv8-classify wrapper for WHO 6-step hand hygiene gesture recognition.

Each frame is classified independently; a short rolling window provides
temporal smoothing via majority vote so noisy single-frame predictions
don't flip the output every frame.
"""

from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np
from ultralytics import YOLO


WHO_STEP_LABELS = {
    0: "palm_to_palm",          # METC code 1 — WHO step 2
    1: "palm_over_dorsum",      # METC code 2 — WHO step 3
    2: "fingers_interlaced",    # METC code 3 — WHO step 4
    3: "backs_of_fingers",      # METC code 4 — WHO step 5
    4: "rotational_thumb",      # METC code 5 — WHO step 6
    5: "fingertips_to_palm",    # METC code 6 — WHO step 7
}


@dataclass
class GesturePrediction:
    """Result for the current smoothed classification."""
    step_id: int
    step_name: str
    confidence: float                  # confidence of the top class on this frame
    all_scores: Dict[str, float]       # {step_name: probability} from this frame


class GestureClassifier:
    """
    Runs YOLOv8-classify on each incoming frame and returns a smoothed
    gesture prediction using a majority-vote rolling window.

    Usage:
        clf = GestureClassifier("models/weights/gesture_classifier.pt")
        clf.push_frame(cropped_frame)
        pred = clf.classify()        # None until window fills
        if pred:
            print(pred.step_name, pred.confidence)
    """

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.6,
        input_size: int = 224,
        smoothing_window: int = 8,     # frames for majority vote
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.smoothing_window = smoothing_window

        self.model = YOLO(weights)
        self.model.to(device)

        # Rolling buffers
        self._pred_buffer: Deque[int] = deque(maxlen=smoothing_window)
        self._last_raw: Optional[GesturePrediction] = None

    # ------------------------------------------------------------------
    # Frame buffer API (matches old LSTM interface so pipeline.py is unchanged)
    # ------------------------------------------------------------------

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Classify `frame` immediately and push the result into the smoothing window.
        `frame` should be a BGR crop of the person / hand region.
        """
        raw = self._classify_single(frame)
        if raw is not None:
            self._pred_buffer.append(raw.step_id)
            self._last_raw = raw

    def ready(self) -> bool:
        """True once the smoothing window is full."""
        return len(self._pred_buffer) >= self.smoothing_window

    def reset(self) -> None:
        self._pred_buffer.clear()
        self._last_raw = None

    # ------------------------------------------------------------------
    # Smoothed classification
    # ------------------------------------------------------------------

    def classify(self) -> Optional[GesturePrediction]:
        """
        Return the majority-voted step for the current window.
        Returns None if the window isn't full yet.
        """
        if not self.ready():
            return None

        majority_id, _ = Counter(self._pred_buffer).most_common(1)[0]

        # Use the raw scores from the most recent frame, but override step_id
        # with the smoothed majority result.
        if self._last_raw is None:
            return None

        return GesturePrediction(
            step_id=majority_id,
            step_name=WHO_STEP_LABELS[majority_id],
            confidence=self._last_raw.confidence,
            all_scores=self._last_raw.all_scores,
        )

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

    def _classify_single(self, frame: np.ndarray) -> Optional[GesturePrediction]:
        """Run YOLOv8-classify on one BGR frame. Returns None below threshold."""
        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            verbose=False,
        )
        if not results:
            return None

        probs = results[0].probs          # ultralytics Probs object
        if probs is None:
            return None

        top1_id   = int(probs.top1)
        top1_conf = float(probs.top1conf)

        if top1_conf < self.conf_threshold:
            return None

        all_scores = {
            WHO_STEP_LABELS[i]: float(probs.data[i])
            for i in range(len(WHO_STEP_LABELS))
            if i < len(probs.data)
        }

        return GesturePrediction(
            step_id=top1_id,
            step_name=WHO_STEP_LABELS.get(top1_id, str(top1_id)),
            confidence=top1_conf,
            all_scores=all_scores,
        )
