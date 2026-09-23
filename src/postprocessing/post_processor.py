"""
Post-Processing Module.
Filters false positives via temporal sliding window smoothing, temporal NMS, cooldown intervals,
and categorizes alerts into confidence tiers (Critical Alert vs Warning).
"""

from collections import deque
from typing import List, Dict, Any, Optional
import numpy as np


class PostProcessor:
    """
    Performs temporal filtering, suppression, and structured incident reporting for accident detections.
    """

    def __init__(
        self,
        temporal_window: int = 5,
        min_consecutive_alerts: int = 3,
        nms_temporal_seconds: float = 2.0,
        cooldown_seconds: float = 5.0,
        high_confidence_thresh: float = 0.80,
        warning_thresh: float = 0.50,
        fps: float = 30.0,
    ):
        self.temporal_window = temporal_window
        self.min_consecutive_alerts = min_consecutive_alerts
        self.nms_frames = int(nms_temporal_seconds * fps)
        self.cooldown_frames = int(cooldown_seconds * fps)
        self.high_thresh = high_confidence_thresh
        self.warning_thresh = warning_thresh
        self.fps = fps

        # Internal state
        # track_id (or pair tuple) -> deque of (frame_id, score, raw_event)
        self.history_buffer = deque(maxlen=self.temporal_window)
        self.confirmed_events: List[Dict[str, Any]] = []
        self.last_alert_frame: int = -9999
        self.event_counter: int = 0

    def process_frame_candidate(
        self,
        frame_id: int,
        candidate_event: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Process detection candidate from current frame.

        Args:
            frame_id: Current video frame index.
            candidate_event: Raw candidate dict containing score, tracks, reasons, bbox, or None.

        Returns:
            Confirmed alert dict if triggered, else None.
        """
        if candidate_event is not None and candidate_event.get("score", 0.0) >= self.warning_thresh:
            self.history_buffer.append((frame_id, candidate_event["score"], candidate_event))
        else:
            self.history_buffer.append((frame_id, 0.0, None))

        # Check if candidate extends an ongoing confirmed incident (e.g. 3rd vehicle joins collision)
        if candidate_event is not None and candidate_event.get("score", 0.0) >= self.warning_thresh and self.confirmed_events:
            last_event = self.confirmed_events[-1]
            time_gap = frame_id - last_event["frame_end"]
            event_duration = frame_id - last_event["frame_start"]
            
            # If within active incident window and not exceeding max event duration (e.g. max 5s)
            if time_gap <= max(self.nms_frames, int(self.fps * 2.0)) and event_duration <= int(self.fps * 5.0):
                c_tracks = set(candidate_event.get("tracks", []))
                last_tracks = set(last_event.get("tracks", []))
                
                # Check track sharing or spatial proximity
                shares_track = bool(c_tracks & last_tracks)
                from ..features.motion_features import compute_iou
                spatial_overlap = False
                if candidate_event.get("bbox") and last_event.get("bbox"):
                    spatial_overlap = compute_iou(np.array(candidate_event["bbox"]), np.array(last_event["bbox"])) > 0.05
                
                if shares_track or spatial_overlap:
                    # Merge and expand ongoing incident
                    merged_tracks = sorted(list(last_tracks | c_tracks))
                    last_event["tracks"] = merged_tracks
                    last_event["num_vehicles"] = len(merged_tracks)
                    last_event["frame_end"] = frame_id
                    last_event["score"] = max(last_event["score"], candidate_event.get("score", 0.0))
                    
                    # Update union bbox
                    cb = candidate_event.get("bbox")
                    lb = last_event.get("bbox")
                    if cb and lb:
                        last_event["bbox"] = [
                            float(min(lb[0], cb[0])),
                            float(min(lb[1], cb[1])),
                            float(max(lb[2], cb[2])),
                            float(max(lb[3], cb[3])),
                        ]
                    self.last_alert_frame = frame_id
                    return None

        # Check if cooldown is active for a separate new event
        if (frame_id - self.last_alert_frame) < self.cooldown_frames:
            return None

        # Check if sliding window has enough consecutive/frequent high-score frames
        valid_frames = [item for item in self.history_buffer if item[1] >= self.warning_thresh]
        if len(valid_frames) >= self.min_consecutive_alerts:
            # Pick highest score event in the window
            best_frame, best_score, best_raw = max(valid_frames, key=lambda x: x[1])

            # Accumulate all unique tracks in the detection window (combines multi-vehicle contacts)
            all_window_tracks = set()
            all_window_boxes = []
            all_window_reasons = []
            for item in valid_frames:
                raw_e = item[2]
                if raw_e:
                    all_window_tracks.update(raw_e.get("tracks", []))
                    if raw_e.get("bbox"):
                        all_window_boxes.append(raw_e["bbox"])
                    if raw_e.get("reasons"):
                        all_window_reasons.append(raw_e["reasons"])

            merged_tracks = sorted(list(all_window_tracks)) if all_window_tracks else best_raw.get("tracks", [])
            num_vehicles = len(merged_tracks)

            # Union bbox over the triggering window
            if all_window_boxes:
                union_box = [
                    float(min(b[0] for b in all_window_boxes)),
                    float(min(b[1] for b in all_window_boxes)),
                    float(max(b[2] for b in all_window_boxes)),
                    float(max(b[3] for b in all_window_boxes)),
                ]
            else:
                union_box = best_raw.get("bbox", [])

            # Determine alert severity level (boosted to CRITICAL if >= 3 vehicles or high score)
            level = "CRITICAL" if (best_score >= self.high_thresh or num_vehicles >= 3) else "WARNING"

            # Check Temporal NMS with existing confirmed events
            if self.confirmed_events:
                last_event = self.confirmed_events[-1]
                if (frame_id - last_event["frame_end"]) < self.nms_frames:
                    # Update previous event window
                    last_event["frame_end"] = frame_id
                    last_event["score"] = max(last_event["score"], best_score)
                    last_event["tracks"] = sorted(list(set(last_event["tracks"]) | set(merged_tracks)))
                    last_event["num_vehicles"] = len(last_event["tracks"])
                    return None

            self.event_counter += 1
            timestamp = frame_id / max(self.fps, 1.0)

            unique_reasons = " | ".join(list(dict.fromkeys(all_window_reasons))) if all_window_reasons else best_raw.get("reasons", "kinematic_anomaly")

            confirmed = {
                "event_id": self.event_counter,
                "frame_start": valid_frames[0][0],
                "frame_end": frame_id,
                "timestamp_sec": round(timestamp, 2),
                "timestamp_str": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}.{int((timestamp % 1) * 10):01d}",
                "level": level,
                "score": round(best_score, 3),
                "tracks": merged_tracks,
                "num_vehicles": num_vehicles,
                "bbox": union_box,
                "reasons": unique_reasons,
            }

            self.confirmed_events.append(confirmed)
            self.last_alert_frame = frame_id
            self.history_buffer.clear()
            return confirmed

        return None

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return list of all registered incidents."""
        return self.confirmed_events
