"""
System Evaluation Script.
Evaluates the accident detection pipeline against annotated ground-truth videos.
Calculates Precision, Recall, F1-Score, False Alarm Rate per hour (FAR/h), and Detection Latency.
"""

import os
import argparse
import json
import time
from typing import Dict, List, Any
import cv2

from src.pipeline import AccidentDetectionPipeline


def evaluate_system(
    manifest_path: str,
    video_dir: str,
    config_path: str = "configs/default.yaml",
    temporal_tolerance_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Run evaluation against ground truth manifest.
    Manifest format:
    [
      {"video_file": "vid1.mp4", "label": "accident", "start_frame": 120, "end_frame": 180},
      {"video_file": "vid2.mp4", "label": "normal", "start_frame": -1, "end_frame": -1}
    ]
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    pipeline = AccidentDetectionPipeline(config_path)

    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    tn = 0  # True Negatives
    latencies = []
    total_video_duration_sec = 0.0
    start_eval_time = time.time()
    total_frames_processed = 0

    print(f"Starting evaluation on {len(ground_truth)} videos...")

    for item in ground_truth:
        vid_file = os.path.join(video_dir, item["video_file"])
        if not os.path.exists(vid_file):
            print(f"⚠️ Video not found: {vid_file}, skipping.")
            continue

        cap = cv2.VideoCapture(vid_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / max(fps, 1.0)
        total_video_duration_sec += duration_sec
        total_frames_processed += frame_count
        cap.release()

        # Run pipeline
        detected_events = pipeline.process_video(vid_file)

        is_actual_accident = (item["label"] == "accident")
        gt_start_sec = item.get("start_frame", 0) / fps

        if is_actual_accident:
            matched = False
            for ev in detected_events:
                ev_time = ev["timestamp_sec"]
                # Check if detection falls within tolerance
                if ev_time >= (gt_start_sec - temporal_tolerance_sec):
                    matched = True
                    latencies.append(max(0.0, ev_time - gt_start_sec))
                    break

            if matched:
                tp += 1
                # Any extra alerts on the same video can be counted as FP if far away
                fp += max(0, len(detected_events) - 1)
            else:
                fn += 1
        else:
            # Normal video
            if len(detected_events) == 0:
                tn += 1
            else:
                fp += len(detected_events)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_hours = total_video_duration_sec / 3600.0
    far_per_hour = fp / total_hours if total_hours > 0 else 0.0
    avg_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0

    elapsed = time.time() - start_eval_time
    avg_fps = total_frames_processed / elapsed if elapsed > 0 else 0.0

    results = {
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "FAR_per_hour": round(far_per_hour, 2),
        "Avg_Detection_Latency_sec": round(avg_latency, 2),
        "Processed_FPS": round(avg_fps, 1),
    }

    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f" {k:<28}: {v}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Accident Detection Pipeline")
    parser.add_argument("--manifest", type=str, default="datasets/vn_traffic/manifest.json", help="Annotations manifest")
    parser.add_argument("--video_dir", type=str, default="datasets/vn_traffic/videos", help="Directory of videos")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Pipeline config")
    args = parser.parse_args()

    if os.path.exists(args.manifest):
        evaluate_system(args.manifest, args.video_dir, args.config)
    else:
        print(f"Manifest not found at {args.manifest}. Create or download annotations first.")
