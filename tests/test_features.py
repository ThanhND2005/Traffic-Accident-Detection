"""
Unit Tests for Motion Feature Extractor Module.
"""

import numpy as np
import pytest
from src.features.motion_features import (
    MotionFeatureExtractor,
    compute_iou,
    moving_average,
)


def test_compute_iou():
    box_a = np.array([0, 0, 10, 10], dtype=np.float32)
    box_b = np.array([0, 0, 10, 10], dtype=np.float32)
    assert compute_iou(box_a, box_b) == pytest.approx(1.0)

    box_c = np.array([10, 10, 20, 20], dtype=np.float32)
    assert compute_iou(box_a, box_c) == pytest.approx(0.0)

    box_d = np.array([5, 0, 15, 10], dtype=np.float32)
    # Inter: 5x10=50. Area A: 100, Area D: 100, Union: 150 -> IoU = 50/150 = 1/3
    assert compute_iou(box_a, box_d) == pytest.approx(1.0 / 3.0)


def test_moving_average():
    data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    smoothed = moving_average(data, window=3)
    assert smoothed.shape == data.shape


def test_single_track_features():
    extractor = MotionFeatureExtractor(smoothing_window=3)
    # Linear movement
    centers = np.array([[i * 10.0, 50.0] for i in range(10)], dtype=np.float32)
    bboxes = np.array([[i * 10, 40, i * 10 + 20, 60] for i in range(10)], dtype=np.float32)

    feat = extractor.compute_single_track_features(centers, bboxes)
    assert "velocity" in feat
    assert "speed" in feat
    assert "acceleration" in feat
    assert "delta_angles" in feat
    assert len(feat["speed"]) == 10


def test_pairwise_sequence():
    extractor = MotionFeatureExtractor(smoothing_window=3)
    track_a = [
        {"frame_id": i, "center": np.array([i * 5.0, 100.0]), "bbox": np.array([i * 5, 90, i * 5 + 20, 110])}
        for i in range(20)
    ]
    track_b = [
        {"frame_id": i, "center": np.array([100.0 - i * 5.0, 100.0]), "bbox": np.array([90 - i * 5, 90, 110 - i * 5, 110])}
        for i in range(20)
    ]

    seq_15d = extractor.compute_pairwise_sequence(track_a, track_b, seq_len=15)
    assert seq_15d is not None
    assert seq_15d.shape == (15, 15)
