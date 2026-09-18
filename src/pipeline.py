"""
Accident Detection Pipeline (Main Orchestrator).
Combines YOLO detection, ByteTrack tracking, trajectory extraction, kinematic analysis,
cascade visual classifier, score fusion, and temporal post-processing.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque
import cv2
import numpy as np
import yaml

from .detection.yolo_detector import YOLODetector
from .tracking.trajectory_manager import TrajectoryManager
from .features.motion_features import MotionFeatureExtractor
from .classifiers.rule_based import RuleBasedAccidentDetector
from .classifiers.lstm_classifier import TrajectoryAccidentClassifier
from .classifiers.video_classifier import VideoAccidentClassifier
from .fusion.fusion import AccidentFusion
from .postprocessing.post_processor import PostProcessor


class AccidentDetectionPipeline:
    """
    End-to-End Pipeline for Automated Traffic Accident Detection.
    """

    def __init__(self, config_path_or_dict: Any = "configs/default.yaml"):
        """
        Initialize the pipeline from a YAML config file or dictionary.
        """
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path_or_dict

        # System settings
        self.device = self.config.get("system", {}).get("device", "cpu")
        self.fps = float(self.config.get("system", {}).get("fps", 30))

        # Module 1: Detector & Tracker
        det_cfg = self.config.get("detector", {})
        trk_cfg = self.config.get("tracker", {})
        self.detector = YOLODetector(
            weights=det_cfg.get("weights", "yolo11s.pt"),
            device=self.device,
            conf_threshold=det_cfg.get("conf_threshold", 0.4),
            iou_threshold=det_cfg.get("iou_threshold", 0.5),
            target_classes=det_cfg.get("classes", [0, 1, 2, 3, 5, 7]),
            tracker_config=trk_cfg.get("config_path", "configs/bytetrack_custom.yaml"),
        )

        # Module 2: Trajectory Manager
        feat_cfg = self.config.get("features", {})
        self.trajectory_manager = TrajectoryManager(
            max_history=feat_cfg.get("max_history", 90)
        )

        # Module 3: Motion Feature Extractor
        self.feature_extractor = MotionFeatureExtractor(
            smoothing_window=feat_cfg.get("smoothing_window", 5)
        )

        # Module 4a: Kinematic Classifiers
        cls_cfg = self.config.get("classifiers", {})
        rule_cfg = cls_cfg.get("rule_based", {})
        self.rule_detector = RuleBasedAccidentDetector(
            sudden_decel_thresh=rule_cfg.get("sudden_decel_thresh", 15.0),
            direction_change_thresh=rule_cfg.get("direction_change_thresh", 45.0),
            iou_overlap_thresh=rule_cfg.get("iou_overlap_thresh", 0.15),
            distance_decrease_rate=rule_cfg.get("distance_decrease_rate", 0.5),
            min_track_length=feat_cfg.get("min_track_length", 10),
            alert_threshold=rule_cfg.get("alert_threshold", 0.5),
        )

        self.use_lstm = cls_cfg.get("lstm", {}).get("enabled", False)
        lstm_weights = cls_cfg.get("lstm", {}).get("weights")
        if self.use_lstm and lstm_weights and os.path.exists(lstm_weights):
            self.lstm_classifier = TrajectoryAccidentClassifier.load_from_checkpoint(
                lstm_weights, device=self.device
            )
        else:
            self.lstm_classifier = None

        # Module 4b: Visual Branch (X3D)
        vis_cfg = cls_cfg.get("visual_x3d", {})
        self.use_visual = vis_cfg.get("enabled", False)
        self.cascade_threshold = vis_cfg.get("cascade_threshold", 0.3)
        self.visual_classifier = VideoAccidentClassifier(
            weights_path=vis_cfg.get("weights"),
            device=self.device,
            num_frames=vis_cfg.get("num_frames", 16),
            crop_size=vis_cfg.get("crop_size", 160),
        ) if self.use_visual else None

        # Module 5: Fusion
        fus_cfg = self.config.get("fusion", {})
        self.fusion = AccidentFusion(
            method=fus_cfg.get("method", "cascade"),
            motion_weight=fus_cfg.get("motion_weight", 0.6),
            visual_weight=fus_cfg.get("visual_weight", 0.4),
            weights_path=fus_cfg.get("weights"),
        )

        # Module 6: Post-Processing
        post_cfg = self.config.get("postprocessing", {})
        self.post_processor = PostProcessor(
            temporal_window=post_cfg.get("temporal_window", 5),
            min_consecutive_alerts=post_cfg.get("min_consecutive_alerts", 3),
            nms_temporal_seconds=post_cfg.get("nms_temporal_seconds", 2.0),
            cooldown_seconds=post_cfg.get("cooldown_seconds", 5.0),
            high_confidence_thresh=post_cfg.get("high_confidence_thresh", 0.8),
            warning_thresh=post_cfg.get("warning_thresh", 0.5),
            fps=self.fps,
        )

        # Recent frame buffer for visual classifier ROI extraction
        self.frame_buffer = deque(maxlen=30)
        self.active_alert_banner: Optional[Dict[str, Any]] = None
        self.banner_display_until_frame: int = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Process a single video frame.

        Returns:
            annotated_frame: Frame with visual tracking and alert overlays.
            confirmed_alert: Alert information if a new incident was confirmed on this frame.
        """
        self.frame_buffer.append(frame.copy())
        annotated = frame.copy()

        # Step 1: Detect & Track
        track_results = self.detector.track(frame, persist=True)
        track_ids = track_results["track_ids"]
        bboxes = track_results["bboxes"]
        classes = track_results["classes"]
        confs = track_results["confs"]

        # Step 2: Update Trajectory History
        self.trajectory_manager.update(frame_idx, track_ids, bboxes, classes, confs)
        self.trajectory_manager.cleanup_old_tracks(frame_idx, max_age=60)

        # Step 3 & 4: Evaluate Candidates (Kinematic & Interaction Analysis)
        candidate_event = None
        highest_score = 0.0

        # Pairwise interaction evaluation
        pairs = self.trajectory_manager.get_interacting_pairs(frame_idx, distance_threshold=150.0)
        for tid_a, tid_b in pairs:
            centers_a = self.trajectory_manager.get_trajectory(tid_a)
            bboxes_a = self.trajectory_manager.get_bboxes(tid_a)
            centers_b = self.trajectory_manager.get_trajectory(tid_b)
            bboxes_b = self.trajectory_manager.get_bboxes(tid_b)

            feat_a = self.feature_extractor.compute_single_track_features(centers_a, bboxes_a)
            feat_b = self.feature_extractor.compute_single_track_features(centers_b, bboxes_b)

            # Kinematic Rule Evaluation
            eval_res = self.rule_detector.evaluate_pairwise(
                tid_a, tid_b, feat_a, feat_b, bboxes_a, bboxes_b, frame_idx
            )

            if eval_res is not None:
                motion_score = eval_res["score"]

                # Optional LSTM scoring
                if self.lstm_classifier is not None:
                    track_dict_a = self.trajectory_manager.tracks[tid_a]
                    track_dict_b = self.trajectory_manager.tracks[tid_b]
                    seq_15d = self.feature_extractor.compute_pairwise_sequence(track_dict_a, track_dict_b)
                    if seq_15d is not None:
                        lstm_score = self.lstm_classifier.predict_proba(seq_15d, device=self.device)
                        motion_score = 0.5 * motion_score + 0.5 * lstm_score

                # Step 4b & 5: Cascade Visual Branch & Fusion
                visual_score = None
                if self.use_visual and self.visual_classifier is not None and motion_score > self.cascade_threshold:
                    clip_frames = list(self.frame_buffer)
                    visual_score = self.visual_classifier.predict(clip_frames)

                fused_score = self.fusion.fuse(motion_score, visual_score)
                eval_res["score"] = fused_score

                if fused_score > highest_score:
                    highest_score = fused_score
                    candidate_event = eval_res

        # Step 6: Temporal Post-Processing
        confirmed_alert = self.post_processor.process_frame_candidate(frame_idx, candidate_event)
        if confirmed_alert is not None:
            self.active_alert_banner = confirmed_alert
            self.banner_display_until_frame = frame_idx + int(self.fps * 3.0)  # Show for 3 sec

        # Step 7: Visual Annotations
        annotated = self._render_visualizations(
            annotated, frame_idx, track_ids, bboxes, candidate_event
        )

        return annotated, confirmed_alert

    def _render_visualizations(
        self,
        frame: np.ndarray,
        frame_idx: int,
        track_ids: np.ndarray,
        bboxes: np.ndarray,
        candidate_event: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Render trajectories, bounding boxes, and alert banners onto the frame."""
        h, w = frame.shape[:2]

        # Draw vehicle tracks
        for i, tid in enumerate(track_ids):
            if tid < 0:
                continue
            box = bboxes[i].astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(
                frame, f"ID:{tid}", (box[0], max(15, box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            # Draw trajectory path
            traj = self.trajectory_manager.get_trajectory(tid)
            if len(traj) > 1:
                pts = traj[-20:].astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255, 200, 0), 2)

        # Highlight collision zone if candidate event present
        if candidate_event is not None and "bbox" in candidate_event:
            cbox = [int(v) for v in candidate_event["bbox"]]
            cv2.rectangle(frame, (cbox[0], cbox[1]), (cbox[2], cbox[3]), (0, 0, 255), 3)

        # Draw active incident banner
        if (
            self.active_alert_banner is not None
            and frame_idx <= self.banner_display_until_frame
        ):
            banner = self.active_alert_banner
            level = banner["level"]
            score = banner["score"]
            reasons = banner["reasons"]

            bg_color = (0, 0, 200) if level == "CRITICAL" else (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (w, 55), bg_color, -1)

            banner_text = f"[{level}] ACCIDENT DETECTED! (Conf: {score:.2f}) | {reasons}"
            cv2.putText(
                frame, banner_text, (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
            )

        # Frame timestamp watermark
        ts_sec = frame_idx / max(self.fps, 1.0)
        info_text = f"Time: {ts_sec:.1f}s | Frame: {frame_idx}"
        cv2.putText(
            frame, info_text, (w - 240, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
        )

        return frame

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process an entire video file from start to finish.

        Args:
            video_path: Path to input video file.
            output_path: Optional path to save annotated output video.
            callback: Optional progress callback fn(frame_idx, total_frames, confirmed_event).

        Returns:
            List of confirmed accident events.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated, confirmed_alert = self.process_frame(frame, frame_idx)

            if writer:
                writer.write(annotated)

            if callback:
                callback(frame_idx, total_frames, confirmed_alert)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        return self.post_processor.get_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Accident Detection CLI")
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, default="runs/output_alert.mp4", help="Output video path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    args = parser.parse_args()

    pipeline = AccidentDetectionPipeline(args.config)
    print(f"[Info] Processing {args.video}...")
    events = pipeline.process_video(args.video, args.output)
    print(f"[Done] Finished processing. Detected {len(events)} incidents.")
    for ev in events:
        print(f" - Incident #{ev['event_id']} at {ev['timestamp_str']} ({ev['level']}, score={ev['score']})")
