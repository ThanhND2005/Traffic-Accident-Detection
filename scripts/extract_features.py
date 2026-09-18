"""
Motion Feature Extraction Batch Script.
Processes video datasets, tracks objects using YOLO11 + ByteTrack, and saves
15-dimensional pairwise interaction feature sequences for LSTM training.
"""

import os
import argparse
import glob
import json
from typing import List
import cv2
import numpy as np
from tqdm import tqdm

from src.detection.yolo_detector import YOLODetector
from src.tracking.trajectory_manager import TrajectoryManager
from src.features.motion_features import MotionFeatureExtractor


def extract_features_from_video(
    video_path: str,
    detector: YOLODetector,
    seq_len: int = 30,
) -> List[np.ndarray]:
    """
    Extract all 15-dimensional pairwise trajectory sequences from a single video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    manager = TrajectoryManager(max_history=90)
    extractor = MotionFeatureExtractor(smoothing_window=5)

    sequences = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        track_res = detector.track(frame, persist=True)
        manager.update(
            frame_idx,
            track_res["track_ids"],
            track_res["bboxes"],
            track_res["classes"],
            track_res["confs"],
        )

        # Inspect interacting pairs
        pairs = manager.get_interacting_pairs(frame_idx, distance_threshold=150.0, min_length=10)
        for tid_a, tid_b in pairs:
            seq = extractor.compute_pairwise_sequence(
                manager.tracks[tid_a],
                manager.tracks[tid_b],
                seq_len=seq_len,
            )
            if seq is not None and len(seq) == seq_len:
                sequences.append(seq)

        frame_idx += 1

    cap.release()
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Extract motion features from videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to video directory")
    parser.add_argument("--output_file", type=str, default="datasets/features/extracted_features.npz", help="Output .npz path")
    parser.add_argument("--weights", type=str, default="yolo11s.pt", help="YOLO model path")
    parser.add_argument("--is_accident", action="store_true", help="Mark all extracted samples as accident (label=1)")
    parser.add_argument("--seq_len", type=int, default=30, help="Temporal sequence length")
    args = parser.parse_args()

    detector = YOLODetector(weights=args.weights, conf_threshold=0.4)
    video_files = glob.glob(os.path.join(args.video_dir, "*.mp4")) + glob.glob(os.path.join(args.video_dir, "*.avi"))
    print(f"Found {len(video_files)} videos in {args.video_dir}")

    all_sequences = []
    all_labels = []
    label_val = 1 if args.is_accident else 0

    for vf in tqdm(video_files, desc="Extracting features"):
        seqs = extract_features_from_video(vf, detector, seq_len=args.seq_len)
        for s in seqs:
            all_sequences.append(s)
            all_labels.append(label_val)

    if len(all_sequences) > 0:
        X = np.stack(all_sequences, axis=0)  # (N, seq_len, 15)
        y = np.array(all_labels, dtype=np.int64)
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        np.savez_compressed(args.output_file, X=X, y=y)
        print(f"✅ Saved {len(X)} sequence samples to {args.output_file} (Shape: {X.shape})")
    else:
        print("⚠️ No valid pairwise sequences extracted.")


if __name__ == "__main__":
    main()
