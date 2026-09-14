"""
Trajectory Management Module.
Maintains history of tracked objects, centroids, bounding boxes, and provides trajectory queries.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class TrajectoryManager:
    """
    Manages spatial-temporal trajectories of all tracked objects across frames.
    """

    def __init__(self, max_history: int = 90):
        """
        Args:
            max_history: Max number of historic states to store per track (e.g. 90 frames ~ 3s at 30 fps).
        """
        self.max_history = max_history
        # track_id -> deque of dict: {"frame_id": int, "bbox": [x1, y1, x2, y2], "center": [cx, cy], "class_id": int, "conf": float}
        self.tracks: Dict[int, deque] = {}
        # track_id -> last observed frame_id
        self.last_seen: Dict[int, int] = {}

    def update(
        self,
        frame_id: int,
        track_ids: np.ndarray,
        bboxes: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
    ):
        """
        Update trajectories with tracking detections from current frame.

        Args:
            frame_id: Integer frame index.
            track_ids: 1D array of track IDs.
            bboxes: 2D array of bounding boxes [x1, y1, x2, y2].
            classes: 1D array of class IDs.
            confs: 1D array of confidence scores.
        """
        for i in range(len(track_ids)):
            tid = int(track_ids[i])
            if tid < 0:
                continue

            bbox = bboxes[i].astype(float)
            cls_id = int(classes[i])
            conf = float(confs[i])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            if tid not in self.tracks:
                self.tracks[tid] = deque(maxlen=self.max_history)

            self.tracks[tid].append({
                "frame_id": frame_id,
                "bbox": bbox,
                "center": np.array([cx, cy], dtype=np.float32),
                "class_id": cls_id,
                "conf": conf,
            })
            self.last_seen[tid] = frame_id

    def get_trajectory(self, track_id: int) -> np.ndarray:
        """
        Get 2D centroids trajectory over time for a given track ID.

        Returns:
            (T, 2) numpy array of [cx, cy] centroids.
        """
        if track_id not in self.tracks or len(self.tracks[track_id]) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        centers = [item["center"] for item in self.tracks[track_id]]
        return np.array(centers, dtype=np.float32)

    def get_bboxes(self, track_id: int) -> np.ndarray:
        """
        Get bounding boxes over time for a track ID.

        Returns:
            (T, 4) numpy array of [x1, y1, x2, y2].
        """
        if track_id not in self.tracks or len(self.tracks[track_id]) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        boxes = [item["bbox"] for item in self.tracks[track_id]]
        return np.array(boxes, dtype=np.float32)

    def get_last_state(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Return the most recent detection state for track_id."""
        if track_id in self.tracks and len(self.tracks[track_id]) > 0:
            return self.tracks[track_id][-1]
        return None

    def get_active_tracks(self, current_frame: int, min_length: int = 10, max_age: int = 5) -> List[int]:
        """
        Get active track IDs that have at least min_length history and were seen recently.

        Args:
            current_frame: Current frame number.
            min_length: Minimum number of historical frames.
            max_age: Maximum frame delay since last observation.

        Returns:
            List of valid track IDs.
        """
        active = []
        for tid, history in self.tracks.items():
            if len(history) >= min_length and (current_frame - self.last_seen[tid]) <= max_age:
                active.append(tid)
        return active

    def cleanup_old_tracks(self, current_frame: int, max_age: int = 60):
        """
        Remove tracks that have not been seen for more than max_age frames.
        """
        dead_tracks = [
            tid for tid, last_f in self.last_seen.items()
            if (current_frame - last_f) > max_age
        ]
        for tid in dead_tracks:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

    def get_interacting_pairs(
        self,
        current_frame: int,
        distance_threshold: float = 120.0,
        min_length: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        Find pairs of active tracks that are geographically close to each other in the current frame.
        Useful for pairwise collision/interaction analysis.
        """
        active_tids = self.get_active_tracks(current_frame, min_length=min_length, max_age=2)
        pairs = []
        n = len(active_tids)
        for i in range(n):
            tid_a = active_tids[i]
            c_a = self.tracks[tid_a][-1]["center"]
            for j in range(i + 1, n):
                tid_b = active_tids[j]
                c_b = self.tracks[tid_b][-1]["center"]
                dist = np.linalg.norm(c_a - c_b)
                if dist <= distance_threshold:
                    pairs.append((tid_a, tid_b))
        return pairs