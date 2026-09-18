"""
notebooks/eda_ccd.py
Run: python notebooks/eda_ccd.py --csv_path path/to/Crash-1500.csv --sep "\t"
"""
from __future__ import annotations
import argparse, os, sys
from typing import Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
from ccd_parser import parse_ccd_csv  # noqa: E402


def run_eda(csv_path: str, sep: Optional[str] = "\t", out_dir: str = "eda_output_ccd") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = parse_ccd_csv(csv_path, sep=sep)

    print(f"\n{'='*60}\nOVERVIEW\n{'='*60}")
    print(f"Total videos: {len(df)}")
    print(f"With accident: {df.has_accident.sum()} ({df.has_accident.mean()*100:.1f}%)")
    print(f"Normal (no accident): {(~df.has_accident).sum()} ({(~df.has_accident).mean()*100:.1f}%)")
    print(">>> CASCADE/FOCAL LOSS NOTE: if the accident/normal ratio is heavily\n"
          "    skewed (e.g. 80/20), this is exactly the basis for deciding how\n"
          "    much focal loss is needed in the tech stack.")

    # 1. Accident/normal ratio (Fix: map directly so pie chart never throws length mismatch if single class)
    fig, ax = plt.subplots(figsize=(5, 5))
    label_map = {True: "Accident", False: "Normal"}
    accident_counts = df.has_accident.map(label_map).value_counts()
    accident_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=["#C44E52", "#4C72B0"][:len(accident_counts)])
    ax.set_ylabel("")
    ax.set_title("Accident vs Normal ratio (CCD)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_accident_vs_normal.png"), dpi=120)
    plt.close()

    # 2. Accident start position within the 5s clip (only for videos with an accident)
    acc_df = df[df.has_accident]
    print(f"\n{'='*60}\n2. ACCIDENT START POSITION (ratio within the 5s clip, accident videos only)\n{'='*60}")
    if len(acc_df) > 0:
        print(acc_df.anomaly_start_ratio.describe())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(acc_df.anomaly_start_ratio, bins=20, color="#C44E52")
        ax.set_title("Accident start position within the 5s clip")
        ax.set_xlabel("anomaly_start_ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "02_anomaly_position.png"), dpi=120)
        plt.close()
    else:
        print("No accident videos found to plot position distribution.")

    print(">>> If the distribution clusters near 1.0 (end of clip): confirms\n"
          "    CCD's known property that 'accidents always fall near the end\n"
          "    of the 5s clip' - this affects sliding window design, unlike\n"
          "    DoTA (where the accident can be anywhere in the video).")

    # 3. Accident segment duration
    print(f"\n{'='*60}\n3. ACCIDENT SEGMENT DURATION (seconds, at 10fps)\n{'='*60}")
    if len(acc_df) > 0:
        acc_df = acc_df.copy()
        acc_df["anomaly_duration_sec"] = acc_df.anomaly_duration_frames / acc_df.src_fps
        print(acc_df.anomaly_duration_sec.describe())
    else:
        print("No accident videos found to calculate duration.")

    # 4. Day/Night
    print(f"\n{'='*60}\n4. RECORDING CONDITIONS\n{'='*60}")
    print("Timing (Day/Night):\n", df.timing.value_counts())
    print("\nWeather:\n", df.weather.value_counts())
    print("\nEgo involve:\n", df.ego_involve.value_counts())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    df.timing.value_counts().plot(kind="bar", ax=axes[0], color="#4C72B0", title="Timing")
    df.weather.value_counts().plot(kind="bar", ax=axes[1], color="#55A868", title="Weather")
    df.ego_involve.value_counts().plot(kind="bar", ax=axes[2], color="#8172B2", title="Ego involve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_conditions_distribution.png"), dpi=120)
    plt.close()

    # 5. Cross-check: does the accident ratio skew by Day/Night or Weather
    #    (cross-imbalance check within CCD)
    print(f"\n{'='*60}\n5. ACCIDENT RATIO BY CONDITION (cross-imbalance check)\n{'='*60}")
    print(pd.crosstab(df.timing, df.has_accident, normalize="index"))

    manifest_path = os.path.join(out_dir, "ccd_video_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest: {manifest_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--sep", type=str, default="\t")
    p.add_argument("--out_dir", type=str, default="eda_output_ccd")
    args = p.parse_args()
    
    # Check if empty string or None passed for sep to allow autodetect
    separator = None if args.sep in ("", "None", "auto") else args.sep
    run_eda(args.csv_path, separator, args.out_dir)