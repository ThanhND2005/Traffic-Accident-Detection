"""
Motion Feature Extractor.
Extracts kinematic and interaction features from spatial-temporal trajectories:
velocities, accelerations, heading angle changes, pairwise IoUs, distances, and area ratios.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) of two boxes [x1, y1, x2, y2].
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def moving_average(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply 1D moving average smoothing along axis 0.
    """
    if len(data) < window or window <= 1:
        return data.copy()
    pad_width = window // 2
    padded = np.pad(data, ((pad_width, pad_width), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.zeros_like(data)
    for dim in range(data.shape[1]):
        conv = np.convolve(padded[:, dim], kernel, mode="valid")
        smoothed[:, dim] = conv[:len(data)]
    return smoothed


class MotionFeatureExtractor:
    """
    Extracts motion dynamics and relational interaction features from tracked trajectories.
    """

    def __init__(self, smoothing_window: int = 5, frame_shape: Optional[Tuple[int, int]] = (720, 1280)):
        """
        Args:
            smoothing_window: Window size for trajectory coordinate smoothing.
            frame_shape: (height, width) for spatial normalization.
        """
        self.smoothing_window = smoothing_window
        self.frame_shape = frame_shape
        if frame_shape is not None:
            self.h, self.w = frame_shape
            self.diag = float(np.sqrt(self.h**2 + self.w**2))
        else:
            self.h, self.w, self.diag = 1.0, 1.0, 1.0

    def compute_single_track_features(self, centers: np.ndarray, bboxes: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute kinematic features for a single track.

        Args:
            centers: (T, 2) array of [cx, cy] centroids.
            bboxes: (T, 4) array of [x1, y1, x2, y2].

        Returns:
            Dictionary containing:
                - 'velocity': (T, 2) [vx, vy]
                - 'speed': (T,) magnitude of velocity
                - 'acceleration': (T, 2) [ax, ay]
                - 'accel_mag': (T,) magnitude of acceleration (deceleration)
                - 'angles': (T,) heading direction in degrees [-180, 180]
                - 'delta_angles': (T,) angular heading change in degrees
                - 'area_ratios': (T,) ratio of current area to previous reference
        """
        T = len(centers)
        if T == 0:
            return {
                "velocity": np.zeros((0, 2), dtype=np.float32),
                "speed": np.zeros((0,), dtype=np.float32),
                "acceleration": np.zeros((0, 2), dtype=np.float32),
                "accel_mag": np.zeros((0,), dtype=np.float32),
                "angles": np.zeros((0,), dtype=np.float32),
                "delta_angles": np.zeros((0,), dtype=np.float32),
                "area_ratios": np.zeros((0,), dtype=np.float32),
            }

        smoothed_centers = moving_average(centers, window=self.smoothing_window)

        # Velocity: v(t) = c(t) - c(t-1)
        velocity = np.zeros((T, 2), dtype=np.float32)
        if T > 1:
            velocity[1:] = smoothed_centers[1:] - smoothed_centers[:-1]
            velocity[0] = velocity[1]  # edge replicate

        speed = np.linalg.norm(velocity, axis=1)

        # Acceleration: a(t) = v(t) - v(t-1)
        acceleration = np.zeros((T, 2), dtype=np.float32)
        if T > 1:
            acceleration[1:] = velocity[1:] - velocity[:-1]
            acceleration[0] = acceleration[1]

        accel_mag = np.linalg.norm(acceleration, axis=1)

        # Heading angle in degrees: arctan2(vy, vx)
        angles = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
        delta_angles = np.zeros((T,), dtype=np.float32)
        if T > 1:
            raw_diff = angles[1:] - angles[:-1]
            # Wrap to [-180, 180]
            wrapped_diff = (raw_diff + 180.0) % 360.0 - 180.0
            delta_angles[1:] = np.abs(wrapped_diff)
            delta_angles[0] = 0.0

        # Bounding box area ratio: area(t) / area(t-k)
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area_ratios = np.ones((T,), dtype=np.float32)
        k = min(5, T - 1)
        if k > 0:
            area_ratios[k:] = areas[k:] / np.maximum(areas[:-k], 1e-3)

        return {
            "velocity": velocity,
            "speed": speed,
            "acceleration": acceleration,
            "accel_mag": accel_mag,
            "angles": angles,
            "delta_angles": delta_angles,
            "area_ratios": area_ratios,
        }

    def compute_pairwise_sequence(
        self,
        track_a_dict: Dict[str, Any],
        track_b_dict: Dict[str, Any],
        seq_len: int = 30,
    ) -> Optional[np.ndarray]:
        """
        Extract the 15-dimensional interaction feature sequence for an interacting pair of tracks
        over the common synchronized temporal window.

        Features per timestep:
        [0..1]: v1_x, v1_y (norm)
        [2..3]: a1_x, a1_y (norm)
        [4]:    delta_angle_1 (/ 180.0)
        [5..6]: v2_x, v2_y (norm)
        [7..8]: a2_x, a2_y (norm)
        [9]:    delta_angle_2 (/ 180.0)
        [10]:   relative distance (norm)
        [11]:   delta distance (convergence rate, norm)
        [12]:   IoU overlap [0..1]
        [13]:   area_ratio_1
        [14]:   area_ratio_2

        Returns:
            (seq_len, 15) numpy array, or None if insufficient common history.
        """
        frames_a = {item["frame_id"]: item for item in track_a_dict}
        frames_b = {item["frame_id"]: item for item in track_b_dict}

        common_frames = sorted(list(set(frames_a.keys()).intersection(frames_b.keys())))
        if len(common_frames) < min(10, seq_len // 2):
            return None

        # Take last seq_len frames
        selected_frames = common_frames[-seq_len:]

        centers_a = np.array([frames_a[f]["center"] for f in selected_frames], dtype=np.float32)
        bboxes_a = np.array([frames_a[f]["bbox"] for f in selected_frames], dtype=np.float32)
        centers_b = np.array([frames_b[f]["center"] for f in selected_frames], dtype=np.float32)
        bboxes_b = np.array([frames_b[f]["bbox"] for f in selected_frames], dtype=np.float32)

        feat_a = self.compute_single_track_features(centers_a, bboxes_a)
        feat_b = self.compute_single_track_features(centers_b, bboxes_b)

        T = len(selected_frames)
        feature_matrix = np.zeros((T, 15), dtype=np.float32)

        # Track A dynamics normalized
        feature_matrix[:, 0:2] = feat_a["velocity"] / (self.diag + 1e-6)
        feature_matrix[:, 2:4] = feat_a["acceleration"] / (self.diag + 1e-6)
        feature_matrix[:, 4] = feat_a["delta_angles"] / 180.0

        # Track B dynamics normalized
        feature_matrix[:, 5:7] = feat_b["velocity"] / (self.diag + 1e-6)
        feature_matrix[:, 7:9] = feat_b["acceleration"] / (self.diag + 1e-6)
        feature_matrix[:, 9] = feat_b["delta_angles"] / 180.0

        # Relational metrics
        diff = centers_a - centers_b
        distances = np.linalg.norm(diff, axis=1)
        feature_matrix[:, 10] = distances / (self.diag + 1e-6)

        delta_dist = np.zeros((T,), dtype=np.float32)
        if T > 1:
            delta_dist[1:] = distances[1:] - distances[:-1]
        feature_matrix[:, 11] = delta_dist / (self.diag + 1e-6)

        # Pairwise IoU
        ious = np.array([compute_iou(bboxes_a[t], bboxes_b[t]) for t in range(T)], dtype=np.float32)
        feature_matrix[:, 12] = ious

        # Area ratios
        feature_matrix[:, 13] = np.clip(feat_a["area_ratios"], 0.0, 5.0) / 5.0
        feature_matrix[:, 14] = np.clip(feat_b["area_ratios"], 0.0, 5.0) / 5.0

        # Pad to seq_len if shorter
        if T < seq_len:
            pad_len = seq_len - T
            padding = np.repeat(feature_matrix[:1], pad_len, axis=0)
            feature_matrix = np.vstack([padding, feature_matrix])

        return feature_matrix
