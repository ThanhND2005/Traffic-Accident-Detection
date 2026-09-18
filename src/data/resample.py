"""
src/data/resample.py

Resample a video to a unified 1 FPS + standardized resolution.

Why this must be done BEFORE frame extraction (instead of "whatever the
dataset gives us"):
- DoTA / CCD / CADP were recorded at different original FPS (typically
  10 - 30 fps).
- If we keep the original FPS, the "motion speed" the model learns gets
  conflated with the filming frame rate, not real physical speed -> motion
  features (velocity, acceleration, IoU convergence...) become
  non-comparable across sources, and the cascade threshold (when to call
  X3D-S) becomes inconsistent.
- Standardizing resolution prevents the model from learning "camera
  characteristics" (object scale differences caused by resolution) instead
  of real accident characteristics.

Resampling strategy: use time-based frame indices (derived from the
original FPS read from the video metadata), NOT "take every Nth frame"
(since N would differ per video depending on its original FPS, which
would be wrong if a single fixed N were applied to every video).
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_FPS_DEFAULT = 1.0
TARGET_RESOLUTION_DEFAULT = (224, 224)  # (width, height)


@dataclass
class ResampleResult:
    video_id: str
    src_path: str
    out_path: str
    src_fps: float
    src_frame_count: int
    src_resolution: tuple
    target_fps: float
    target_resolution: tuple
    out_frame_count: int
    duration_sec: float
    status: str  # "ok" | "skipped" | "error"
    error_msg: Optional[str] = None


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    return cap


def get_video_metadata(path: str) -> dict:
    """Read real metadata of a video (fps, frame count, resolution, duration)."""
    cap = _open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps and fps > 0 else 0.0
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration,
        }
    finally:
        cap.release()


def resample_video_to_frames(
    src_path: str,
    video_id: str,
    out_dir: str,
    target_fps: float = TARGET_FPS_DEFAULT,
    target_resolution: tuple = TARGET_RESOLUTION_DEFAULT,
    resize_mode: str = "letterbox",
    save_as: str = "frames",  # "frames" | "video"
) -> ResampleResult:
    """
    Resample 1 video to target_fps + target_resolution.

    resize_mode:
      - "letterbox": keep the original aspect ratio, pad with black bars
        (like YOLO does) -> avoids distortion, recommended since the
        pipeline includes YOLO11 detection.
      - "stretch": resize directly, distorts aspect ratio but simpler.

    save_as:
      - "frames": save each frame as .jpg into out_dir/video_id/frame_%05d.jpg
      - "video": save as a new .mp4 file at out_dir/video_id.mp4
    """
    try:
        meta = get_video_metadata(src_path)
        src_fps = meta["fps"]
        if not src_fps or src_fps <= 0:
            return ResampleResult(
                video_id=video_id, src_path=src_path, out_path="",
                src_fps=src_fps, src_frame_count=meta["frame_count"],
                src_resolution=(meta["width"], meta["height"]),
                target_fps=target_fps, target_resolution=target_resolution,
                out_frame_count=0, duration_sec=meta["duration_sec"],
                status="error", error_msg="Source FPS <= 0 or metadata unreadable",
            )

        cap = _open_video(src_path)
        src_frame_count = meta["frame_count"]
        duration_sec = meta["duration_sec"]

        # Timestamps (seconds) to sample at target_fps, from 0 -> duration
        n_target_frames = max(1, int(np.floor(duration_sec * target_fps)) + 1)
        timestamps = np.arange(n_target_frames) / target_fps
        timestamps = timestamps[timestamps <= duration_sec + 1e-6]

        # Convert timestamp -> frame index using the ORIGINAL fps (must be
        # rounded per-video since original fps differs across videos)
        frame_indices = np.round(timestamps * src_fps).astype(int)
        frame_indices = np.clip(frame_indices, 0, max(src_frame_count - 1, 0))
        frame_indices = sorted(set(frame_indices.tolist()))

        out_frames = []
        target_w, target_h = target_resolution

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _resize_frame(frame, target_w, target_h, resize_mode)
            out_frames.append(frame)
        cap.release()

        if len(out_frames) == 0:
            return ResampleResult(
                video_id=video_id, src_path=src_path, out_path="",
                src_fps=src_fps, src_frame_count=src_frame_count,
                src_resolution=(meta["width"], meta["height"]),
                target_fps=target_fps, target_resolution=target_resolution,
                out_frame_count=0, duration_sec=duration_sec,
                status="error", error_msg="Could not read any frame (video may be corrupted)",
            )

        os.makedirs(out_dir, exist_ok=True)
        if save_as == "frames":
            video_out_dir = os.path.join(out_dir, video_id)
            os.makedirs(video_out_dir, exist_ok=True)
            for i, frame in enumerate(out_frames):
                cv2.imwrite(os.path.join(video_out_dir, f"frame_{i:05d}.jpg"), frame)
            out_path = video_out_dir
        elif save_as == "video":
            out_path = os.path.join(out_dir, f"{video_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, target_fps, (target_w, target_h))
            for frame in out_frames:
                writer.write(frame)
            writer.release()
        else:
            raise ValueError(f"Invalid save_as: {save_as}")

        return ResampleResult(
            video_id=video_id, src_path=src_path, out_path=out_path,
            src_fps=src_fps, src_frame_count=src_frame_count,
            src_resolution=(meta["width"], meta["height"]),
            target_fps=target_fps, target_resolution=target_resolution,
            out_frame_count=len(out_frames), duration_sec=duration_sec,
            status="ok",
        )

    except Exception as e:  # noqa: BLE001
        logger.exception(f"Error while resampling {src_path}")
        return ResampleResult(
            video_id=video_id, src_path=src_path, out_path="",
            src_fps=0.0, src_frame_count=0, src_resolution=(0, 0),
            target_fps=target_fps, target_resolution=target_resolution,
            out_frame_count=0, duration_sec=0.0,
            status="error", error_msg=str(e),
        )


def _resize_frame(frame: np.ndarray, target_w: int, target_h: int, mode: str) -> np.ndarray:
    if mode == "stretch":
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    if mode == "letterbox":
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        return canvas

    raise ValueError(f"Invalid resize_mode: {mode}")


def save_resample_log(results: list, out_json_path: str) -> None:
    """Save resample results log (for auditing & debugging later)."""
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    n_ok = sum(1 for r in results if r.status == "ok")
    n_err = sum(1 for r in results if r.status == "error")
    logger.info(f"Resample log: {n_ok} succeeded, {n_err} failed. Details: {out_json_path}")