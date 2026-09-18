"""
src/data/ccd_parser.py

Parses the REAL annotation of CCD (Car Crash Dataset, Bao et al. 2020, ACM MM 2020).

CSV format (confirmed from the user's file + cross-checked with the
official repo Cogito2012/CarCrashDataset):
  vidname | frame_1 ... frame_50 (called 'binlabels' in the official repo) | startframe | youtubeID | timing | weather | egoinvolve

- frame_1..frame_50: BINARY label (0/1) for EACH frame within a fixed
  50-frame clip. 0 = normal, 1 = accident in progress.
- startframe: the (zero-padded) frame index in the ORIGINAL, untrimmed
  YouTube video where this 5s clip starts - NOT the frame where the
  accident starts within the clip.
- timing: Day/Night. weather: Normal/Rainy/Snowy/... . egoinvolve: Yes/No.

CONFIRMED FACTS (no longer an assumption) - cross-checked against 6 papers
+ the official author repo:
- CCD = 4,500 videos (1,500 positive + 3,000 negative), each video 5
  seconds, 50 frames => FIXED 10 FPS. Author's standard split: 3,600 train
  / 900 test (80/20), positive:negative ratio = 1:2 preserved in both sets.
- If present, the accident ALWAYS falls within the LAST 2 SECONDS of the
  5s clip (DIFFERENT from DoTA, where the accident can be anywhere in the
  video). This directly affects EDA section (2) and how sliding
  windows/cascade thresholds should be designed for CCD videos.

IMPORTANT - NEW FINDING TO FOLLOW UP ON:
- The 1,500 POSITIVE videos (with an accident) were sourced from YouTube.
- The 3,000 NEGATIVE videos (normal) were sourced from **BDD100K** (a
  COMPLETELY DIFFERENT source, per the official repo) - NOT the same
  source as the positive videos.
- This is very likely why you only see 1 image folder 'CrashBest' (prefix
  'C_' = Crash, positive videos) next to the CSV, and NO image folder for
  negative videos: images for the 3,000 normal videos may (a) not have
  been downloaded yet, or (b) live in a separate source/folder derived
  from BDD100K that does NOT follow the 'C_xxxxxx_xx.jpg' naming
  convention. NEEDS TO BE CONFIRMED WITH THE TEAM before writing the image
  loading logic for negative videos.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CCD_SOURCE_FPS = 10.0        # CONFIRMED (no longer an assumption) - see docstring
CCD_NUM_FRAMES = 50          # fixed by the CCD format (5s clip @ 10fps)
FRAME_COLS = [f"frame_{i}" for i in range(1, CCD_NUM_FRAMES + 1)]


@dataclass
class CcdVideoRecord:
    video_id: str
    dataset_source: str
    youtube_id: str
    num_frames: int
    has_accident: bool
    accident_category: str      # unified schema with dota_parser ('crash' or 'normal')
    anomaly_start: int          # 0-indexed, -1 if no accident (normal video)
    anomaly_end: int            # 0-indexed, -1 if no accident
    anomaly_duration_frames: int
    anomaly_start_ratio: float
    timing: str                 # Day / Night
    weather: str
    ego_involve: bool
    src_fps: float


def parse_ccd_csv(csv_path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Reads the original CCD annotation CSV, returns a VIDEO-LEVEL manifest.
    sep: if None, automatically infers delimiter (e.g. '\\t' or ','); otherwise uses sep."""
    if sep is None:
        raw = pd.read_csv(csv_path, sep=None, engine="python", dtype={"vidname": str})
    else:
        raw = pd.read_csv(csv_path, sep=sep, dtype={"vidname": str})

    missing_cols = set(FRAME_COLS) - set(raw.columns)
    if missing_cols:
        raise ValueError(
            f"Missing frame columns in CSV: {sorted(missing_cols)[:5]}... "
            f"Check whether the 'sep' argument is correct (the file might use ',' instead of tab)."
        )

    records = []
    for _, row in raw.iterrows():
        frame_labels = row[FRAME_COLS].astype(int).values
        accident_idx = [i for i, v in enumerate(frame_labels) if v == 1]  # 0-indexed

        if accident_idx:
            a_start, a_end = min(accident_idx), max(accident_idx)
            has_accident = True
        else:
            a_start, a_end = -1, -1
            has_accident = False

        duration = (a_end - a_start + 1) if has_accident else 0
        records.append(CcdVideoRecord(
            video_id=str(row["vidname"]),
            dataset_source="ccd",
            youtube_id=str(row.get("youtubeID", "")),
            num_frames=CCD_NUM_FRAMES,
            has_accident=has_accident,
            accident_category="crash" if has_accident else "normal",
            anomaly_start=a_start,
            anomaly_end=a_end,
            anomaly_duration_frames=duration,
            anomaly_start_ratio=(a_start / CCD_NUM_FRAMES) if has_accident else -1.0,
            timing=str(row.get("timing", "unknown")).strip(),
            weather=str(row.get("weather", "unknown")).strip(),
            ego_involve=(str(row.get("egoinvolve", "")).strip().lower() == "yes"),
            src_fps=CCD_SOURCE_FPS,
        ))

    df = pd.DataFrame([r.__dict__ for r in records])
    n_pos = df.has_accident.sum()
    logger.info(f"Read {len(df)} CCD videos: {n_pos} with an accident, {len(df) - n_pos} normal.")

    dup = df[df.duplicated(subset=["video_id"], keep=False)]
    if len(dup) > 0:
        logger.warning(
            f"WARNING: {dup.video_id.nunique()} video_id values are duplicated in the "
            f"source CSV (e.g. {dup.video_id.unique()[:5].tolist()}) - clean up before splitting."
        )
    return df


def build_ccd_frame_manifest(video_df: pd.DataFrame, raw_csv_path: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Returns a FRAME-LEVEL manifest (analogous to
    dota_parser.build_dota_frame_manifest) - needed for the stride-based
    frame resampling step."""
    if sep is None:
        raw = pd.read_csv(raw_csv_path, sep=None, engine="python", dtype={"vidname": str})
    else:
        raw = pd.read_csv(raw_csv_path, sep=sep, dtype={"vidname": str})

    rows = []
    for _, row in raw.iterrows():
        vid = str(row["vidname"])
        for i, col in enumerate(FRAME_COLS):  # i = 0-indexed frame_id
            rows.append({
                "video_id": vid,
                "frame_id": i,
                "frame_label": "accident" if int(row[col]) == 1 else "normal",
                "is_anomaly_frame": int(row[col]) == 1,
            })
    return pd.DataFrame(rows)


def guess_frame_image_path(video_id: str, frame_id: int, frames_root: str,
                           positive_prefix: str = "C") -> str:
    """
    CONFIRMED from the actual Drive screenshot (folder 'ccd/CrashBest/'):
    filenames look like C_000001_01.jpg -> C_<video_id 6 digits>_<frame_idx 2 digits>.jpg

    - video_id is zero-padded to 6 digits (vidname="1" -> "000001").
    - frame_idx in the FILENAME is 1-INDEXED (01..50), different from this
      module's INTERNAL frame_id (0-indexed, 0..49) - this function
      automatically adds 1 when building the filename.
    - positive_prefix="C" (Crash) - ASSUMED to be the prefix specifically
      for videos WITH an accident, living in the 'CrashBest' folder. There
      is likely a SEPARATE folder + prefix for 'normal' videos (e.g.
      'N_000001_01.jpg' in a differently-named folder) - NEEDS
      CONFIRMATION: check Drive for another folder next to 'CrashBest'
      (commonly named 'Normal' or 'Negative'), and whether its images use
      an 'N_' prefix. If so, pass positive_prefix="N" when calling this
      function for videos with has_accident=False.
    """
    frame_idx_1indexed = frame_id + 1
    filename = f"{positive_prefix}_{int(video_id):06d}_{frame_idx_1indexed:02d}.jpg"
    return os.path.join(frames_root, filename)