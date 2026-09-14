"""
Data Preprocessing Script.
Extracts frames from video clips, creates YOLO dataset splits, and crops spatio-temporal video snippets.
"""

import os
import argparse
import glob
import cv2
import numpy as np


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_rate: int = 2,
    prefix: str = "frame",
) -> int:
    """
    Extract frames at specified sampling rate (frames per second).
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to read {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps / frame_rate))

    count = 0
    saved = 0
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            out_file = os.path.join(output_dir, f"{base_name}_{prefix}_{saved:05d}.jpg")
            cv2.imwrite(out_file, frame)
            saved += 1
        count += 1

    cap.release()
    return saved


def slice_accident_clip(
    video_path: str,
    output_clip_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: int = 16,
):
    """
    Slice a short segment from a video for X3D visual training.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    curr = start_frame
    while curr <= end_frame and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        curr += 1
    cap.release()

    if len(frames) > 0:
        os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_clip_path, fourcc, target_fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"✅ Saved clip: {output_clip_path} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video datasets")
    parser.add_argument("--video_dir", type=str, default="datasets/vn_traffic/videos/accident", help="Input video folder")
    parser.add_argument("--output_dir", type=str, default="datasets/vn_traffic/images/train", help="Frame output folder")
    parser.add_argument("--sample_rate", type=int, default=2, help="Sample frames per second")
    args = parser.parse_args()

    videos = glob.glob(f"{args.video_dir}/*.mp4") + glob.glob(f"{args.video_dir}/*.avi")
    print(f"Found {len(videos)} videos in {args.video_dir}")
    total_saved = 0
    for v in videos:
        saved = extract_frames_from_video(v, args.output_dir, frame_rate=args.sample_rate)
        total_saved += saved
    print(f"Finished. Extracted {total_saved} frames to {args.output_dir}.")
