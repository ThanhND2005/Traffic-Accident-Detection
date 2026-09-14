"""
Rule-Based Accident Detector (MVP Baseline).
Applies heuristic kinematic and spatial interaction rules to detect vehicle collisions.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class RuleBasedAccidentDetector:
    """
    Evaluates kinematic rules (sudden deceleration, acute trajectory angular deviation,
    sudden IoU overlap, rapid mutual convergence) to compute an accident anomaly score.
    """

    def __init__(
        self,
        sudden_decel_thresh: float = 15.0,
        direction_change_thresh: float = 45.0,
        iou_overlap_thresh: float = 0.15,
        distance_decrease_rate: float = 0.50,
        min_track_length: int = 10,
        alert_threshold: float = 0.50,
    ):
        self.sudden_decel_thresh = sudden_decel_thresh
        self.direction_change_thresh = direction_change_thresh
        self.iou_overlap_thresh = iou_overlap_thresh
        self.distance_decrease_rate = distance_decrease_rate
        self.min_track_length = min_track_length
        self.alert_threshold = alert_threshold

    def evaluate_pairwise(
        self,
        tid_a: int,
        tid_b: int,
        feat_a: Dict[str, np.ndarray],
        feat_b: Dict[str, np.ndarray],
        bboxes_a: np.ndarray,
        bboxes_b: np.ndarray,
        current_frame: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate interaction between two tracks at the current frame.
        """
        reasons = []
        rule_count = 0

        # Rule 1: Sudden Deceleration on either vehicle
        decel_a = feat_a["accel_mag"][-1] if len(feat_a["accel_mag"]) > 0 else 0.0
        decel_b = feat_b["accel_mag"][-1] if len(feat_b["accel_mag"]) > 0 else 0.0

        if decel_a > self.sudden_decel_thresh or decel_b > self.sudden_decel_thresh:
            rule_count += 1
            reasons.append(f"sudden_decel(a={decel_a:.1f}, b={decel_b:.1f})")

        # Rule 2: Sudden Direction Change
        d_angle_a = feat_a["delta_angles"][-1] if len(feat_a["delta_angles"]) > 0 else 0.0
        d_angle_b = feat_b["delta_angles"][-1] if len(feat_b["delta_angles"]) > 0 else 0.0

        if d_angle_a > self.direction_change_thresh or d_angle_b > self.direction_change_thresh:
            rule_count += 1
            reasons.append(f"direction_change(a={d_angle_a:.1f}°, b={d_angle_b:.1f}°)")

        # Rule 3: Pairwise Bounding Box Overlap
        box_a = bboxes_a[-1]
        box_b = bboxes_b[-1]
        from ..features.motion_features import compute_iou
        iou = compute_iou(box_a, box_b)
        if iou >= self.iou_overlap_thresh:
            rule_count += 1
            reasons.append(f"iou_overlap({iou:.2f})")

        # Rule 4: Rapid Convergence
        # If tracks were approaching each other rapidly over the last 5 frames
        k = min(5, len(bboxes_a), len(bboxes_b))
        if k >= 3:
            c_a_now = (box_a[:2] + box_a[2:]) / 2.0
            c_b_now = (box_b[:2] + box_b[2:]) / 2.0
            dist_now = np.linalg.norm(c_a_now - c_b_now)

            c_a_prev = (bboxes_a[-k][:2] + bboxes_a[-k][2:]) / 2.0
            c_b_prev = (bboxes_b[-k][:2] + bboxes_b[-k][2:]) / 2.0
            dist_prev = np.linalg.norm(c_a_prev - c_b_prev)

            if dist_prev > 1.0:
                rel_drop = (dist_prev - dist_now) / dist_prev
                if rel_drop >= self.distance_decrease_rate:
                    rule_count += 1
                    reasons.append(f"rapid_convergence({rel_drop * 100:.0f}%)")

        if rule_count == 0:
            return None

        # Scoring scheme from engineering plan:
        # Base: 0.25 per violated rule
        # Multiplier: 1 rule = 0.25, 2 rules = 0.50 * 1.5 = 0.75, 3+ rules = 0.75+ * 2.0 = 1.0 (clamped)
        raw_score = rule_count * 0.25
        if rule_count == 2:
            score = raw_score * 1.5
        elif rule_count >= 3:
            score = raw_score * 2.0
        else:
            score = raw_score

        score = float(np.clip(score, 0.0, 1.0))

        if score < self.alert_threshold:
            return None

        # Calculate joint union bounding box for visual highlighting
        union_bbox = [
            float(min(box_a[0], box_b[0])),
            float(min(box_a[1], box_b[1])),
            float(max(box_a[2], box_b[2])),
            float(max(box_a[3], box_b[3])),
        ]

        return {
            "frame_id": current_frame,
            "tracks": [tid_a, tid_b],
            "score": score,
            "rules_triggered": rule_count,
            "reasons": " + ".join(reasons),
            "bbox": union_bbox,
        }

    def detect_single(
        self,
        tid: int,
        feat: Dict[str, np.ndarray],
        bbox: np.ndarray,
        current_frame: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate single-track anomalies (e.g., solo motorcycle fall or single vehicle rollover).
        """
        reasons = []
        rule_count = 0

        decel = feat["accel_mag"][-1] if len(feat["accel_mag"]) > 0 else 0.0
        d_angle = feat["delta_angles"][-1] if len(feat["delta_angles"]) > 0 else 0.0
        area_ratio = feat["area_ratios"][-1] if len(feat["area_ratios"]) > 0 else 1.0

        if decel > self.sudden_decel_thresh * 1.2:
            rule_count += 1
            reasons.append(f"solo_sudden_decel({decel:.1f})")

        if d_angle > self.direction_change_thresh * 1.3:
            rule_count += 1
            reasons.append(f"solo_spin({d_angle:.1f}°)")

        if area_ratio < 0.4 or area_ratio > 2.5:  # Sudden aspect change (vehicle flipped/fallen)
            rule_count += 1
            reasons.append(f"deformation_or_fall(ratio={area_ratio:.2f})")

        if rule_count >= 2:
            score = float(np.clip(rule_count * 0.35, 0.0, 1.0))
            if score >= self.alert_threshold:
                return {
                    "frame_id": current_frame,
                    "tracks": [tid],
                    "score": score,
                    "rules_triggered": rule_count,
                    "reasons": " + ".join(reasons),
                    "bbox": bbox.tolist(),
                }
        return None
