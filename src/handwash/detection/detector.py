"""
Layer 1 — Object Detection
YOLOv8 detection of persons, sinks, and soap dispensers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detected object in a frame."""
    class_id: int
    class_name: str
    bbox: np.ndarray           # [x1, y1, x2, y2] in pixel coords
    confidence: float


class HandwashDetector:
    """
    Wraps a YOLOv8 model to detect persons, sinks, and soap dispensers.

    Usage:
        detector = HandwashDetector("models/weights/yolov8_detection.pt")
        detections = detector.detect(frame)
    """

    CLASS_NAMES = {0: "person", 1: "sink", 2: "soap_dispenser"}

    def __init__(
        self,
        weights: str,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.device = device

        self.model = YOLO(weights)
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame (as returned by OpenCV).

        Returns:
            List of Detection objects for all classes above confidence threshold.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.CLASS_NAMES.get(class_id, str(class_id)),
                        bbox=box.xyxy.squeeze().cpu().numpy(),
                        confidence=float(box.conf.item()),
                    )
                )
        return detections

    def filter_by_class(self, detections: List[Detection], class_name: str) -> List[Detection]:
        """Helper: return only detections of a specific class."""
        return [d for d in detections if d.class_name == class_name]
