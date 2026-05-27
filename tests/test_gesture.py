"""Tests for gesture classifier (model architecture + frame buffer)."""

import numpy as np
import pytest
import torch


class TestGestureModel:
    def test_model_output_shape(self):
        from handwash.gesture.classifier import GestureModel
        model = GestureModel(num_classes=7)
        model.eval()
        x = torch.zeros(2, 8, 3, 224, 224)   # batch=2, seq=8
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7)

    def test_frame_buffer_ready(self):
        from handwash.gesture.classifier import GestureClassifier
        clf = GestureClassifier.__new__(GestureClassifier)
        clf.sequence_length = 4
        clf._frame_buffer = []
        clf.conf_threshold = 0.0

        assert not clf.ready()
        for _ in range(4):
            clf._frame_buffer.append(np.zeros((224, 224, 3), dtype=np.uint8))
        assert clf.ready()

    def test_frame_buffer_sliding_window(self):
        from handwash.gesture.classifier import GestureClassifier
        clf = GestureClassifier.__new__(GestureClassifier)
        clf.sequence_length = 4
        clf._frame_buffer = []

        for i in range(6):
            clf.push_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        assert len(clf._frame_buffer) == 4   # sliding window, not growing
