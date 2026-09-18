"""
src/data/dota_parser.py

Parses the REAL annotation schema of DoTA (confirmed from the JSON file the
user provided + the official docs at MoonBlvd/Detection-of-Traffic-Anomaly):

{
  "video_name": str,          # = video_id
  "channel": str,
  "num_frames": int,
  "ignore": bool,
  "ego_involve": bool,
  "night": bool,
  "anomaly_start": int,       # frame_id where the anomaly starts
  "anomaly_end": int,         # frame_id where the anomaly ends
  "video_start": int,
  "video_end": int,
  "accident_id": str,
  "accident_name": str,       # accident type (e.g. "leave_to_left") - video-level
  "labels": [                 # 1 entry / frame
     {"frame_id": int, "image_path": str, "accident_id": int,
      "accident_name": str,  # "normal" or accident type name - frame-level
      "objects": [...]}
  ]
}

IMPORTANT - CONFIRMED FACTS:
- Frames are pre-extracted at a FIXED 10 FPS (video2frames.py -f 10), not
  the original YouTube video's FPS. Therefore we do NOT need/cannot read
  FPS metadata from a source video (the video files haven't even been
  downloaded). Resampling to 1 FPS = take 1 frame out of every 10 frames.
- Every DoTA video has exactly one anomaly window (there is no "fully
  normal" video in this schema) - the "normal" portion is the frames
  BEFORE/AFTER anomaly_start/end within the SAME video. So when stratifying
  the split, use accident_name (video-level) as the label, NOT a binary
  accident/normal label (since virtually 100% of videos would be "accident").
"""

from __future__ import annotations

import os
import json
import glob
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOTA_SOURCE_FPS = 10.0  # confirmed, not a guess


@dataclass
class DotaVideoRecord:
    video_id: str
    dataset_source: str
    channel: str
    num_frames: int
    ignore: bool
    ego_involve: bool
    night: bool
    anomaly_start: int
    anomaly_end: int
    accident_category: str          # video-level accident_name (used for stratification)
    anomaly_duration_frames: int
    anomaly_start_ratio: float      # anomaly start position / total frames (0..1)
    anomaly_duration_ratio: float   # anomaly duration ratio / total video length
    src_fps: float
    annotation_path: str


def parse_dota_annotation(json_path: str) -> Optional[DotaVideoRecord]:
    """Reads one DoTA .json annotation file, returns a video-level summary
    record. Returns None if the video has 'ignore': true (per DoTA convention)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Could not read {json_path}: {e}")
        return None

    if data.get("ignore", False):
        logger.info(f"Skipping {data.get('video_name')} because ignore=true")
        return None

    num_frames = data["num_frames"]
    a_start = data["anomaly_start"]
    a_end = data["anomaly_end"]
    duration = max(0, a_end - a_start + 1)
    # Nếu a_start <= 0 hoặc a_end < a_start (video bình thường/lỗi), duration = 0
    if a_end >= a_start > 0:
        duration = a_end - a_start + 1
        start_ratio = a_start / num_frames if num_frames > 0 else 0.0
    else:
        duration = 0
        start_ratio = 0.0

    return DotaVideoRecord(
        video_id=data["video_name"],
        dataset_source="dota",
        channel=data.get("channel", "unknown"),
        num_frames=num_frames,
        ignore=data.get("ignore", False),
        ego_involve=data.get("ego_involve", False),
        night=data.get("night", False),
        anomaly_start=a_start,
        anomaly_end=a_end,
        accident_category=data.get("accident_name", "unknown"),
        anomaly_duration_frames=duration,
        anomaly_start_ratio=(a_start / num_frames) if num_frames > 0 else 0.0,
        anomaly_duration_ratio=(duration / num_frames) if num_frames > 0 else 0.0,
        src_fps=DOTA_SOURCE_FPS,
        annotation_path=json_path,
    )


def build_dota_manifest(annotations_dir: str) -> pd.DataFrame:
    """Scans all *.json files in annotations_dir, returns a DataFrame (1 row /
    video). This is the VIDEO-LEVEL manifest - used for splitting and overall
    EDA. To get the FRAME-LEVEL manifest (needed for resampling + training),
    use `build_dota_frame_manifest` below."""
    json_files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {annotations_dir}")

    records = []
    n_ignored = 0
    for jf in json_files:
        rec = parse_dota_annotation(jf)
        if rec is None:
            n_ignored += 1
            continue
        records.append(rec)

    df = pd.DataFrame([r.__dict__ for r in records])
    logger.info(
        f"Read {len(json_files)} annotation files: {len(df)} valid videos, "
        f"{n_ignored} videos skipped (ignore=true or read error)."
    )
    return df


def build_dota_frame_manifest(annotations_dir: str, frames_root: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a FRAME-LEVEL manifest: each row is 1 frame, with video_id,
    frame_id, image_path (relative path as stored in the annotation), label
    (that SPECIFIC frame's accident_name - 'normal' or an accident type
    name), plus the absolute path + whether the image file actually exists
    on disk (if frames_root is provided and the frames have been downloaded).
    """
    json_files = sorted(glob.glob(os.path.join(annotations_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {annotations_dir}")

    rows = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("ignore", False):
            continue
        video_id = data["video_name"]
        for lab in data["labels"]:
            row = {
                "video_id": video_id,
                "frame_id": lab["frame_id"],
                "image_path_rel": lab["image_path"],
                "frame_label": lab["accident_name"],
                "is_anomaly_frame": lab["accident_name"] != "normal",
            }
            if frames_root:
                abs_path = os.path.join(frames_root, lab["image_path"])
                row["image_path_abs"] = abs_path
                row["image_exists"] = os.path.exists(abs_path)
            rows.append(row)

    df = pd.DataFrame(rows)
    if frames_root:
        n_missing = (~df["image_exists"]).sum() if len(df) else 0
        if n_missing > 0:
            logger.warning(
                f"{n_missing}/{len(df)} frames have NO matching image file at {frames_root} "
                f"(frames may not be fully downloaded yet)."
            )
    else:
        logger.info(
            "frames_root not provided -> returning manifest from annotations only, "
            "not checking whether image files actually exist."
        )
    return df