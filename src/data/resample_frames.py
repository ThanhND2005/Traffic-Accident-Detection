"""
src/data/resample_frames.py

Resamples DoTA/CCD to a UNIFIED 1 FPS.

Unlike resample.py (written for raw video read via cv2.VideoCapture), this
file is for datasets that have ALREADY been split into frames (DoTA and CCD:
frames pre-extracted at a fixed 10 FPS - see dota_parser.py / ccd_parser.py).
So resampling here simply means SELECTING 1 FRAME OUT OF EVERY STRIDE
CONSECUTIVE FRAMES (stride = src_fps / target_fps = 10 / 1 = 10), no
video reading/seeking needed.

IMPORTANT - handling anomaly frames when downsampling:
If we only take frames 0, 10, 20, 30... on a fixed grid, there's a risk of
"skipping over" an entire anomaly segment if it's shorter than 1 stride and
falls between 2 grid points.
=> Strategy: take the fixed grid (evenly spaced, guarantees a well-defined
   FPS) as the BASE, PLUS add the first and last frame of the
   anomaly_start/anomaly_end window (if not already in the grid) to
   guarantee we never completely miss a short accident event. These added
   frames are marked separately (is_grid_frame=False) so downstream code
   (e.g. clipping for X3D-S) can distinguish "evenly-sampled frames" from
   "frames force-added to avoid losing signal".
"""

from __future__ import annotations

import os
import shutil
import logging
from typing import Optional

import cv2
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_RESOLUTION_DEFAULT = (224, 224)


def select_resampled_frames(
    video_frame_df: pd.DataFrame,
    anomaly_start: int,
    anomaly_end: int,
    src_fps: float = 10.0,
    target_fps: float = 1.0,
) -> pd.DataFrame:
    """
    video_frame_df: frame-level manifest FOR 1 VIDEO ('frame_id' column
                     required), sorted ascending by frame_id.
    Returns the selected subset of frames, with an added 'is_grid_frame' column.
    """
    stride = max(1, round(src_fps / target_fps))
    max_frame_id = int(video_frame_df["frame_id"].max())

    grid_ids = set(range(0, max_frame_id + 1, stride))
    # Guarantee no anomaly signal shorter than 1 stride gets lost
    forced_ids = set()
    if anomaly_end >= anomaly_start >= 0:
        if anomaly_start not in grid_ids:
            forced_ids.add(anomaly_start)
        if anomaly_end not in grid_ids:
            forced_ids.add(anomaly_end)

    selected_ids = grid_ids | forced_ids
    out = video_frame_df[video_frame_df["frame_id"].isin(selected_ids)].copy()
    out["is_grid_frame"] = out["frame_id"].isin(grid_ids)
    out = out.sort_values("frame_id").reset_index(drop=True)
    return out


def resize_and_save_frame(
    src_image_path: str,
    dst_image_path: str,
    target_resolution: tuple = TARGET_RESOLUTION_DEFAULT,
    resize_mode: str = "letterbox",
) -> bool:
    """Reads one real image, resizes it to a standard resolution, saves to
    dst. Returns False if the source file doesn't exist."""
    if not os.path.exists(src_image_path):
        return False
    img = cv2.imread(src_image_path)
    if img is None:
        return False

    target_w, target_h = target_resolution
    if resize_mode == "stretch":
        out_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    elif resize_mode == "letterbox":
        import numpy as np
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target_h, target_w, 3), dtype="uint8")
        top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized
        out_img = canvas
    else:
        raise ValueError(f"Invalid resize_mode: {resize_mode}")

    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
    cv2.imwrite(dst_image_path, out_img)
    return True


def resample_dataset(
    frame_manifest_df: pd.DataFrame,
    video_manifest_df: pd.DataFrame,
    frames_root: Optional[str],
    out_root: str,
    src_fps: float = 10.0,
    target_fps: float = 1.0,
    target_resolution: tuple = TARGET_RESOLUTION_DEFAULT,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Runs resampling for the WHOLE dataset.

    dry_run=True: only computes which frames would be selected (does not
    read/write real images) - use this while images haven't been downloaded
    yet (current situation: video/frames not available). Once frames_root
    exists with real images, set dry_run=False to actually resize + save new
    images into out_root.

    Returns: a log DataFrame (1 row / video) for auditing.
    """
    logs = []
    video_ids = video_manifest_df["video_id"].tolist()

    for video_id in video_ids:
        vrow = video_manifest_df[video_manifest_df.video_id == video_id].iloc[0]
        vframes = frame_manifest_df[frame_manifest_df.video_id == video_id].sort_values("frame_id")
        if len(vframes) == 0:
            logs.append({"video_id": video_id, "status": "error",
                         "msg": "No frames found in manifest", "n_selected": 0})
            continue

        selected = select_resampled_frames(
            vframes, anomaly_start=vrow["anomaly_start"], anomaly_end=vrow["anomaly_end"],
            src_fps=src_fps, target_fps=target_fps,
        )

        n_saved, n_missing = 0, 0
        if not dry_run and frames_root:
            for _, frow in selected.iterrows():
                src = os.path.join(frames_root, frow["image_path_rel"])
                dst = os.path.join(out_root, video_id, os.path.basename(frow["image_path_rel"]))
                ok = resize_and_save_frame(src, dst, target_resolution)
                if ok:
                    n_saved += 1
                else:
                    n_missing += 1

        logs.append({
            "video_id": video_id,
            "status": "ok" if len(selected) > 0 else "error",
            "n_selected": len(selected),
            "n_saved": n_saved,
            "n_missing_images": n_missing,
            "dry_run": dry_run,
        })

    log_df = pd.DataFrame(logs)
    logger.info(
        f"Finished resampling {len(log_df)} videos. "
        f"{'(DRY RUN - no real images read/written)' if dry_run else f'Saved images to {out_root}'}"
    )
    if not dry_run and log_df.get("n_missing_images", pd.Series(dtype=int)).sum() > 0:
        logger.warning(
            f"Total {log_df['n_missing_images'].sum()} frames had no matching image file - "
            f"check whether frames_root has been fully downloaded."
        )
    return log_df