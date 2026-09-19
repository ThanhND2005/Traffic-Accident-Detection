"""
Unit Tests for Tracking & Trajectory Manager Module.
"""

import numpy as np
import pytest
from src.tracking.trajectory_manager import TrajectoryManager


def test_trajectory_manager_update():
    manager = TrajectoryManager(max_history=10)

    # Frame 1
    tids = np.array([1, 2], dtype=np.int32)
    bboxes = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=np.float32)
    classes = np.array([2, 3], dtype=np.int32)
    confs = np.array([0.9, 0.85], dtype=np.float32)

    manager.update(0, tids, bboxes, classes, confs)
    assert 1 in manager.tracks
    assert 2 in manager.tracks
    assert len(manager.tracks[1]) == 1

    # Check trajectory centroid
    traj1 = manager.get_trajectory(1)
    assert traj1.shape == (1, 2)
    assert np.allclose(traj1[0], [30.0, 30.0])


def test_trajectory_interacting_pairs():
    manager = TrajectoryManager(max_history=10)

    # Place two tracks close to each other
    tids = np.array([10, 11], dtype=np.int32)
    bboxes = np.array([[50, 50, 90, 90], [60, 60, 100, 100]], dtype=np.float32)
    classes = np.array([3, 3], dtype=np.int32)
    confs = np.array([0.9, 0.9], dtype=np.float32)

    for f in range(5):
        manager.update(f, tids, bboxes, classes, confs)

    pairs = manager.get_interacting_pairs(4, distance_threshold=50.0, min_length=3)
    assert len(pairs) == 1
    assert pairs[0] == (10, 11)


def test_trajectory_cleanup():
    manager = TrajectoryManager(max_history=10)
    manager.update(0, np.array([1]), np.array([[0, 0, 10, 10]]), np.array([2]), np.array([0.9]))

    assert 1 in manager.tracks
    # Advance time without updating track 1
    manager.cleanup_old_tracks(current_frame=100, max_age=50)
    assert 1 not in manager.tracks


def test_trajectory_active_and_all_tracks():
    manager = TrajectoryManager(max_history=50)
    for f in range(10):
        manager.update(f, np.array([1]), np.array([[0, 0, 10, 10]]), np.array([2]), np.array([0.9]))
    # Now simulate time moving to frame 100 without track 1
    manager.current_frame = 100

    # With default max_age=30, track 1 is inactive at frame 100
    assert len(manager.get_active_tracks(min_length=5)) == 0
    # With max_age=None, all tracks meeting min_length are returned
    assert manager.get_active_tracks(min_length=5, max_age=None) == [1]
    # get_all_tracks returns all tracks meeting min_length
    assert manager.get_all_tracks(min_length=5) == [1]

