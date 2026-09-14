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

        # Check if cooldown is active
        if (frame_id - self.last_alert_frame) < self.cooldown_frames:
            return None

        # Check if sliding window has enough consecutive/frequent high-score frames
        valid_frames = [item for item in self.history_buffer if item[1] >= self.warning_thresh]
        if len(valid_frames) >= self.min_consecutive_alerts:
            # Pick highest score event in the window
            best_frame, best_score, best_raw = max(valid_frames, key=lambda x: x[1])

            # Determine alert severity level
            level = "CRITICAL" if best_score >= self.high_thresh else "WARNING"

            # Check Temporal NMS with existing confirmed events
            if self.confirmed_events:
                last_event = self.confirmed_events[-1]
                if (frame_id - last_event["frame_end"]) < self.nms_frames:
                    # Update previous event window
                    last_event["frame_end"] = frame_id
                    last_event["score"] = max(last_event["score"], best_score)
                    return None

            self.event_counter += 1
            timestamp = frame_id / max(self.fps, 1.0)

            confirmed = {
                "event_id": self.event_counter,
                "frame_start": valid_frames[0][0],
                "frame_end": frame_id,
                "timestamp_sec": round(timestamp, 2),
                "timestamp_str": f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}.{int((timestamp % 1) * 10):01d}",
                "level": level,
                "score": round(best_score, 3),
                "tracks": best_raw.get("tracks", []),
                "bbox": best_raw.get("bbox", []),
                "reasons": best_raw.get("reasons", "kinematic_anomaly"),
            }

            self.confirmed_events.append(confirmed)
            self.last_alert_frame = frame_id
            self.history_buffer.clear()
            return confirmed

        return None

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return list of all registered incidents."""
        return self.confirmed_events
