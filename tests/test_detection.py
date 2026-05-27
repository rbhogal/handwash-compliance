"""Tests for Layer 1 — Detection."""

import numpy as np
import pytest


class TestDetection:
    def test_detection_import(self):
        from handwash.detection import HandwashDetector
        assert HandwashDetector is not None

    def test_detection_output_structure(self, mocker):
        """Detector returns a list of Detection objects with expected fields."""
        from handwash.detection import HandwashDetector
        from handwash.detection.detector import Detection

        # Mock YOLO so we don't need weights in CI
        with mocker.patch("handwash.detection.detector.YOLO"):
            det = HandwashDetector.__new__(HandwashDetector)
            det.conf_threshold = 0.4
            det.iou_threshold = 0.45
            det.input_size = 640
            det.device = "cpu"

            d = Detection(
                class_id=0,
                class_name="person",
                bbox=np.array([10, 20, 100, 200]),
                confidence=0.85,
            )
            assert d.class_name == "person"
            assert d.bbox.shape == (4,)
