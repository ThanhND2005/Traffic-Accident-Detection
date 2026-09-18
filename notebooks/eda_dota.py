"""
notebooks/eda_dota.py

EDA on the DoTA VIDEO-LEVEL manifest (only needs the annotation JSON files,
does NOT need the real images downloaded - all of this analysis is
computed from metadata/labels).

Run: python notebooks/eda_dota.py --annotations_dir path/to/dota/annotations
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, save straight to file
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
from dota_parser import build_dota_manifest  # noqa: E402


def run_eda(annotations_dir: str, out_dir: str = "eda_output") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = build_dota_manifest(annotations_dir)

    print(f"\n{'='*60}\nOVERVIEW\n{'='*60}")
    print(f"Total valid videos: {len(df)}")
    print(f"Total frames (before resampling): {df.num_frames.sum():,}")
    print(f"Number of source channels: {df.channel.nunique()}")

    # 1. Accident category distribution (decides whether focal loss / class
    #    weighting is needed)
    print(f"\n{'='*60}\n1. ACCIDENT CATEGORY DISTRIBUTION\n{'='*60}")
    cat_counts = df.accident_category.value_counts()
    print(cat_counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_title("Accident type distribution (accident_category) - video level")
    ax.set_ylabel("Number of videos")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_category_distribution.png"), dpi=120)
    plt.close()

    # 2. Video length (decides window size for BiLSTM / clip length for X3D-S)
    print(f"\n{'='*60}\n2. VIDEO LENGTH (num_frames @ original 10fps)\n{'='*60}")
    print(df.num_frames.describe())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df.num_frames, bins=30, color="#55A868")
    ax.set_title("Frame count per video distribution (original 10 FPS)")
    ax.set_xlabel("num_frames")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_video_length_distribution.png"), dpi=120)
    plt.close()

    # 3. Where accidents occur in the timeline (decides sliding window design)
    print(f"\n{'='*60}\n3. ACCIDENT START POSITION IN VIDEO (ratio 0..1)\n{'='*60}")
    print(df.anomaly_start_ratio.describe())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df.anomaly_start_ratio, bins=20, color="#C44E52")
    ax.set_title("Accident start position / total video length")
    ax.set_xlabel("anomaly_start_ratio (0=start of video, 1=end of video)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_anomaly_position_distribution.png"), dpi=120)
    plt.close()

    # 4. Accident duration (seconds, converted using original 10fps)
    print(f"\n{'='*60}\n4. ACCIDENT DURATION (seconds, at original 10fps)\n{'='*60}")
    df["anomaly_duration_sec"] = df.anomaly_duration_frames / df.src_fps
    print(df.anomaly_duration_sec.describe())
    n_short = (df.anomaly_duration_sec < 1.0).sum()
    print(f"\n>>> CASCADE WARNING: {n_short}/{len(df)} videos have an accident "
          f"segment SHORTER THAN 1 SECOND. If resampled to 1 FPS without "
          f"forcing boundary frames to be kept (already handled in "
          f"resample_frames.py), these cases would be COMPLETELY LOST.")

    # 5. Night / day, ego_involve ratio (affects domain gap, augmentation)
    print(f"\n{'='*60}\n5. RECORDING CONDITIONS (NIGHT / EGO_INVOLVE)\n{'='*60}")
    print(f"night=True ratio: {df.night.mean()*100:.1f}%")
    print(f"ego_involve=True ratio: {df.ego_involve.mean()*100:.1f}%")

    # 6. Distribution by source channel (checks domain gap across channels)
    print(f"\n{'='*60}\n6. DISTRIBUTION BY CHANNEL\n{'='*60}")
    print(df.channel.value_counts())

    # Save the processed manifest so later steps (split, real resampling)
    # can reuse it without re-parsing all the JSON files
    manifest_path = os.path.join(out_dir, "dota_video_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved video-level manifest: {manifest_path}")
    print(f"Charts saved in: {out_dir}/")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="eda_output")
    args = parser.parse_args()
    run_eda(args.annotations_dir, args.out_dir)