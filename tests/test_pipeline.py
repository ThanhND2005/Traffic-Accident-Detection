"""
Unit & Integration Tests for AccidentDetectionPipeline.
"""

import numpy as np
import pytest
from src.pipeline import AccidentDetectionPipeline


def test_pipeline_initialization():
    pipeline = AccidentDetectionPipeline("configs/default.yaml")
    assert pipeline is not None
    assert pipeline.detector is not None
    assert pipeline.trajectory_manager is not None
    assert pipeline.feature_extractor is not None
    assert pipeline.rule_detector is not None
    assert pipeline.post_processor is not None


def test_pipeline_process_frame():
    pipeline = AccidentDetectionPipeline("configs/default.yaml")
    synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated, alert = pipeline.process_frame(synthetic_frame, frame_idx=0)
    assert annotated.shape == synthetic_frame.shape
    # No accident on a black frame
    assert alert is None


def test_pipeline_simulated_collision():
    pipeline = AccidentDetectionPipeline("configs/default.yaml")

    # Manually inject two tracks approaching each other violently to trigger rule detector
    tids = np.array([1, 2], dtype=np.int32)
    classes = np.array([3, 3], dtype=np.int32)
    confs = np.array([0.9, 0.9], dtype=np.float32)

    # Frame 0 to 4: approaching
    # Frame 5: sudden overlap and stop
    for f in range(6):
        if f < 5:
            # Vehicle 1 moving right
            b1 = np.array([100 + f * 20, 200, 140 + f * 20, 240], dtype=np.float32)
            # Vehicle 2 moving left
            b2 = np.array([300 - f * 20, 200, 340 - f * 20, 240], dtype=np.float32)
        else:
            # Collision: sudden high overlap and stopped
            b1 = np.array([190, 200, 230, 240], dtype=np.float32)
            b2 = np.array([195, 200, 235, 240], dtype=np.float32)

        bboxes = np.stack([b1, b2], axis=0)
        pipeline.trajectory_manager.update(f, tids, bboxes, classes, confs)

    # Check that rule detector flags collision
    c_a = pipeline.trajectory_manager.get_trajectory(1)
    b_a = pipeline.trajectory_manager.get_bboxes(1)
    c_b = pipeline.trajectory_manager.get_trajectory(2)
    b_b = pipeline.trajectory_manager.get_bboxes(2)

    f_a = pipeline.feature_extractor.compute_single_track_features(c_a, b_a)
    f_b = pipeline.feature_extractor.compute_single_track_features(c_b, b_b)

    eval_res = pipeline.rule_detector.evaluate_pairwise(1, 2, f_a, f_b, b_a, b_b, current_frame=5)
    assert eval_res is not None
    assert eval_res["score"] >= 0.5
