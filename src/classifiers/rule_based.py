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
        sudden_decel_thresh: float = 2.0,
        direction_change_thresh: float = 25.0,
        iou_overlap_thresh: float = 0.05,
        distance_decrease_rate: float = 0.25,
        min_track_length: int = 4,
        alert_threshold: float = 0.30,
        min_speed_for_turn: float = 1.5,
        decel_drop_ratio: float = 0.30,
        **kwargs,
    ):
        self.sudden_decel_thresh = sudden_decel_thresh
        self.direction_change_thresh = direction_change_thresh
        self.iou_overlap_thresh = iou_overlap_thresh
        self.distance_decrease_rate = distance_decrease_rate
        self.min_track_length = min_track_length
        self.alert_threshold = alert_threshold
        self.min_speed_for_turn = min_speed_for_turn
        self.decel_drop_ratio = decel_drop_ratio

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
        Applies speed-gated turn analysis, drop-ratio deceleration, and spatial contact rules.
        """
        reasons = []
        rule_score = 0.0
        rules_triggered = 0

        # Lấy lịch sử tốc độ
        speed_a = feat_a.get("speed", np.zeros((0,)))
        speed_b = feat_b.get("speed", np.zeros((0,)))
        speed_a_cur = speed_a[-1] if len(speed_a) > 0 else 0.0
        speed_b_cur = speed_b[-1] if len(speed_b) > 0 else 0.0
        speed_a_prev = np.max(speed_a[-5:-1]) if len(speed_a) > 1 else speed_a_cur
        speed_b_prev = np.max(speed_b[-5:-1]) if len(speed_b) > 1 else speed_b_cur

        # Cả 2 xe đứng yên từ trước -> bỏ qua (xe đỗ hoặc dừng đèn đỏ cạnh nhau)
        if max(speed_a_prev, speed_b_prev) < 1.5 and max(speed_a_cur, speed_b_cur) < 1.0:
            return None

        # Rule 1: Giảm tốc đột ngột (Sudden Deceleration)
        drop_a = speed_a_prev - speed_a_cur
        drop_b = speed_b_prev - speed_b_cur
        decel_a_mag = feat_a["accel_mag"][-1] if len(feat_a.get("accel_mag", [])) > 0 else 0.0
        decel_b_mag = feat_b["accel_mag"][-1] if len(feat_b.get("accel_mag", [])) > 0 else 0.0

        decel_triggered = False
        if (speed_a_prev >= 2.2 and (drop_a >= self.sudden_decel_thresh or (drop_a / max(speed_a_prev, 1e-3)) >= self.decel_drop_ratio)) or (decel_a_mag > self.sudden_decel_thresh * 1.5):
            decel_triggered = True
            reasons.append(f"decel_a({speed_a_prev:.1f}->{speed_a_cur:.1f})")
        if (speed_b_prev >= 2.2 and (drop_b >= self.sudden_decel_thresh or (drop_b / max(speed_b_prev, 1e-3)) >= self.decel_drop_ratio)) or (decel_b_mag > self.sudden_decel_thresh * 1.5):
            decel_triggered = True
            reasons.append(f"decel_b({speed_b_prev:.1f}->{speed_b_cur:.1f})")

        if decel_triggered:
            rules_triggered += 1
            rule_score += 0.35

        # Rule 2: Đổi hướng đột ngột (Speed-Gated Heading Deviation / Spin do va chạm)
        d_angle_a = feat_a["delta_angles"][-1] if len(feat_a.get("delta_angles", [])) > 0 else 0.0
        d_angle_b = feat_b["delta_angles"][-1] if len(feat_b.get("delta_angles", [])) > 0 else 0.0
        turn_triggered = False

        if speed_a_prev >= self.min_speed_for_turn and d_angle_a > self.direction_change_thresh:
            turn_triggered = True
            reasons.append(f"spin_a({d_angle_a:.0f}°)")
        if speed_b_prev >= self.min_speed_for_turn and d_angle_b > self.direction_change_thresh:
            turn_triggered = True
            reasons.append(f"spin_b({d_angle_b:.0f}°)")

        if turn_triggered:
            rules_triggered += 1
            rule_score += 0.30

        # Rule 3: Speed Collapse — xe đang chạy nhanh đột ngột dừng hẳn
        speed_collapse = False
        if speed_a_prev >= 2.5 and speed_a_cur < 0.6:
            speed_collapse = True
            reasons.append(f"collapse_a({speed_a_prev:.1f}->0)")
        if speed_b_prev >= 2.5 and speed_b_cur < 0.6:
            speed_collapse = True
            reasons.append(f"collapse_b({speed_b_prev:.1f}->0)")
        if speed_collapse:
            rules_triggered += 1
            rule_score += 0.30

        # Rule 4: Bounding box tiếp xúc / chồng lấn (Contact & IoU Overlap)
        box_a = bboxes_a[-1]
        box_b = bboxes_b[-1]
        from ..features.motion_features import compute_iou
        iou = compute_iou(box_a, box_b)
        c_a_now = (box_a[:2] + box_a[2:]) / 2.0
        c_b_now = (box_b[:2] + box_b[2:]) / 2.0
        dist_now = float(np.linalg.norm(c_a_now - c_b_now))
        
        contact_triggered = False
        if iou >= 0.12 and max(speed_a_prev, speed_b_prev) >= 1.5:
            contact_triggered = True
            rules_triggered += 1
            rule_score += 0.40
            reasons.append(f"iou_overlap({iou:.2f})")
        elif iou >= 0.05 and (decel_triggered or turn_triggered or speed_collapse):
            contact_triggered = True
            rules_triggered += 1
            rule_score += 0.25
            reasons.append(f"contact({iou:.2f})")

        # Rule 5: Tiến lại gần với tốc độ cao (Rapid Convergence)
        k = min(5, len(bboxes_a), len(bboxes_b))
        if k >= 3:
            c_a_prev = (bboxes_a[-k][:2] + bboxes_a[-k][2:]) / 2.0
            c_b_prev = (bboxes_b[-k][:2] + bboxes_b[-k][2:]) / 2.0
            dist_prev = float(np.linalg.norm(c_a_prev - c_b_prev))

            if dist_prev > 1.0:
                rel_drop = (dist_prev - dist_now) / dist_prev
                if rel_drop >= self.distance_decrease_rate and max(speed_a_prev, speed_b_prev) >= 2.5:
                    rules_triggered += 1
                    rule_score += 0.25
                    reasons.append(f"converge({rel_drop * 100:.0f}%)")

        # Điều kiện bắt buộc cho va chạm 2 xe:
        # PHẢI có va chạm tiếp xúc (contact/iou) HOẶC khoảng cách rất gần (<70px) kèm giảm tốc/sụp đổ tốc độ
        has_physical_contact = contact_triggered or (dist_now < 70.0 and (decel_triggered or speed_collapse))
        if not has_physical_contact:
            return None

        # Yêu cầu tối thiểu 2 dấu hiệu vi phạm để loại bỏ nhiễu thao tác lái thông thường
        if rules_triggered < 2:
            return None

        score = float(np.clip(rule_score * 1.15, 0.0, 1.0))
        if score < self.alert_threshold:
            return None

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
            "rules_triggered": rules_triggered,
            "reasons": " + ".join(reasons),
            "bbox": union_bbox,
        }

    def cluster_multi_vehicle_candidates(
        self,
        pairwise_candidates: List[Dict[str, Any]],
        current_items: List[Dict[str, Any]],
        current_frame: int,
    ) -> List[Dict[str, Any]]:
        """
        Merge pairwise collision candidates into multi-vehicle collision clusters (e.g. 3+ vehicles).
        Groups vehicles that share tracks or are in immediate physical contact during the collision.
        """
        if not pairwise_candidates:
            return []

        from collections import defaultdict
        from ..features.motion_features import compute_iou

        # Build adjacency graph between tracks
        adj = defaultdict(set)
        pair_map = defaultdict(list)
        all_candidate_tracks = set()

        for c in pairwise_candidates:
            c_tr = c.get("tracks", [])
            for t in c_tr:
                all_candidate_tracks.add(t)
                pair_map[t].append(c)
            if len(c_tr) >= 2:
                for i in range(len(c_tr)):
                    for j in range(i + 1, len(c_tr)):
                        adj[c_tr[i]].add(c_tr[j])
                        adj[c_tr[j]].add(c_tr[i])

        # Find connected components (collision clusters)
        visited = set()
        components = []
        for tid in all_candidate_tracks:
            if tid not in visited:
                comp = set()
                queue = [tid]
                visited.add(tid)
                while queue:
                    curr = queue.pop(0)
                    comp.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        items_by_tid = {it["track_id"]: it for it in current_items}
        clustered_results = []

        for comp in components:
            comp_tracks = set(comp)
            comp_reasons = []
            max_score = 0.0
            boxes = []

            for tid in comp:
                for c in pair_map[tid]:
                    max_score = max(max_score, c["score"])
                    comp_reasons.append(c["reasons"])
                    boxes.append(c["bbox"])

            # Compute unified bounding box for the cluster
            cluster_bbox = [
                float(min(b[0] for b in boxes)),
                float(min(b[1] for b in boxes)),
                float(max(b[2] for b in boxes)),
                float(max(b[3] for b in boxes)),
            ]

            # Check if any other vehicle in the current frame is directly touching/overlapping this cluster
            for it in current_items:
                tid = it["track_id"]
                if tid not in comp_tracks:
                    b = np.array(it["bbox"], dtype=np.float32)
                    cb = np.array(cluster_bbox, dtype=np.float32)
                    if compute_iou(b, cb) > 0.08:
                        comp_tracks.add(tid)
                        comp_reasons.append(f"colliding_vehicle(#{tid})")
                        cluster_bbox = [
                            float(min(cluster_bbox[0], b[0])),
                            float(min(cluster_bbox[1], b[1])),
                            float(max(cluster_bbox[2], b[2])),
                            float(max(cluster_bbox[3], b[3])),
                        ]

            num_vehicles = len(comp_tracks)
            # Bonus score for multi-vehicle collisions (3+ vehicles)
            if num_vehicles >= 3:
                final_score = min(1.0, max_score + 0.15)
                comp_reasons.append(f"multi_vehicle_crash({num_vehicles}_vehicles)")
            else:
                final_score = max_score

            unique_reasons = " | ".join(list(dict.fromkeys(comp_reasons)))
            clustered_results.append({
                "frame_id": current_frame,
                "tracks": sorted(list(comp_tracks)),
                "num_vehicles": num_vehicles,
                "score": round(float(final_score), 3),
                "rules_triggered": len(comp_reasons),
                "reasons": unique_reasons,
                "bbox": cluster_bbox,
            })

        return clustered_results

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

        speed = feat.get("speed", np.zeros((0,)))
        speed_cur = speed[-1] if len(speed) > 0 else 0.0
        speed_prev = np.max(speed[-5:-1]) if len(speed) > 1 else speed_cur
        drop = speed_prev - speed_cur

        decel = feat["accel_mag"][-1] if len(feat.get("accel_mag", [])) > 0 else 0.0
        d_angle = feat["delta_angles"][-1] if len(feat.get("delta_angles", [])) > 0 else 0.0
        area_ratio = feat["area_ratios"][-1] if len(feat.get("area_ratios", [])) > 0 else 1.0

        # Cần tốc độ ban đầu đáng kể (>= 3.0 px/frame)
        if speed_prev < 3.0:
            return None

        reasons = []
        rule_count = 0
        if drop >= 3.0 and speed_cur < 0.6:
            rule_count += 1
            reasons.append(f"solo_stop({speed_prev:.1f}->0)")

        # Spin mạnh do mất lái / va đập (> 55 độ)
        if d_angle > 55.0:
            rule_count += 1
            reasons.append(f"solo_spin({d_angle:.0f}°)")

        # Lật xe / ngã xe (aspect ratio thay đổi bất thường)
        if area_ratio < 0.35 or area_ratio > 2.8:
            rule_count += 1
            reasons.append(f"solo_fall(ratio={area_ratio:.2f})")

        # Phải thỏa mãn ít nhất 2 điều kiện (ngã + dừng, hoặc xoay gắt + dừng hẳn)
        if rule_count >= 2:
            return {
                "frame_id": current_frame,
                "tracks": [tid],
                "num_vehicles": 1,
                "score": 0.75,
                "rules_triggered": rule_count,
                "reasons": " + ".join(reasons),
                "bbox": bbox.tolist(),
            }
        return None
