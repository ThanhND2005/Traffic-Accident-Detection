"""
YOLO11 Detector and Tracker Wrapper.
Integrates Ultralytics YOLO with ByteTrack for reliable traffic participant detection.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch


class YOLODetector:
    """
    Wrapper around Ultralytics YOLO11 model for object detection and multi-object tracking.
    """

    def __init__(
        self,
        weights: str = "yolo11s.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        target_classes: Optional[List[int]] = None,
        tracker_config: str = "configs/bytetrack_custom.yaml",
    ):
        """
        Initialize the YOLO detector.

        Args:
            weights: Path to YOLO weights (.pt) or model name (e.g., 'yolo11s.pt').
            device: Device string ('cuda', 'cpu', etc.). If None, auto-detects.
            conf_threshold: Minimum confidence score for detection.
            iou_threshold: NMS IoU threshold.
            target_classes: List of class IDs to filter (e.g. vehicles, pedestrians).
            tracker_config: Path to the ByteTrack yaml config file.
        """
        self.weights = weights
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        self.tracker_config = tracker_config

        # Lazy loading or immediate load of YOLO
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the Ultralytics YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            self.model.to(self.device)
        except Exception as e:
            # Keep reference for mocking/graceful degradation if ultralytics not installed or offline
            print(f"[Warning] Failed to initialize YOLO with '{self.weights}': {e}. Detector in dummy mode.")
            self.model = None

    def detect(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run raw detection on a single frame.

        Args:
            frame: BGR numpy image frame.

        Returns:
            Dictionary containing:
                - 'bboxes': (N, 4) ndarray [x1, y1, x2, y2]
                - 'confs': (N,) ndarray of confidence scores
                - 'classes': (N,) ndarray of integer class IDs
        """
        if self.model is None:
            return {
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "confs": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            }

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "confs": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            }

        return {
            "bboxes": boxes.xyxy.cpu().numpy(),
            "confs": boxes.conf.cpu().numpy(),
            "classes": boxes.cls.cpu().numpy().astype(np.int32),
        }

    def track(self, frame: np.ndarray, persist: bool = True) -> Dict[str, np.ndarray]:
        """
        Run tracking with ByteTrack on a video frame.

        Args:
            frame: BGR numpy image frame.
            persist: Keep tracks alive across consecutive frames.

        Returns:
            Dictionary containing:
                - 'track_ids': (N,) ndarray of integer track IDs
                - 'bboxes': (N, 4) ndarray [x1, y1, x2, y2]
                - 'confs': (N,) ndarray of confidence scores
                - 'classes': (N,) ndarray of integer class IDs
        """
        if self.model is None:
            return {
                "track_ids": np.zeros((0,), dtype=np.int32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "confs": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            }

        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {
                "track_ids": np.zeros((0,), dtype=np.int32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "confs": np.zeros((0,), dtype=np.float32),
                "classes": np.zeros((0,), dtype=np.int32),
            }

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(np.int32)
        else:
            track_ids = np.array([-1] * len(boxes), dtype=np.int32)

        return {
            "track_ids": track_ids,
            "bboxes": boxes.xyxy.cpu().numpy(),
            "confs": boxes.conf.cpu().numpy(),
            "classes": boxes.cls.cpu().numpy().astype(np.int32),
        }
