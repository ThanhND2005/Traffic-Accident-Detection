"""
Unit Tests for YOLO Detector & Tracker Module.
"""

import numpy as np
import pytest
from src.detection.yolo_detector import YOLODetector


def test_detector_initialization():
    """Test detector instantiation with dummy or default settings."""
    detector = YOLODetector(weights="yolo11s.pt", device="cpu")
    assert detector is not None
    assert detector.conf_threshold == 0.4
    assert detector.iou_threshold == 0.5


def test_detect_synthetic_frame():
    """Test detection on a blank synthetic image."""
    detector = YOLODetector(weights="yolo11s.pt", device="cpu")
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.detect(synthetic_frame)
    assert "bboxes" in res
    assert "confs" in res
    assert "classes" in res
    assert isinstance(res["bboxes"], np.ndarray)


def test_track_synthetic_frame():
    """Test tracking on a blank synthetic image."""
    detector = YOLODetector(weights="yolo11s.pt", device="cpu")
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.track(synthetic_frame, persist=True)
    assert "track_ids" in res
    assert "bboxes" in res
    assert "confs" in res
    assert "classes" in res
    assert isinstance(res["track_ids"], np.ndarray)
