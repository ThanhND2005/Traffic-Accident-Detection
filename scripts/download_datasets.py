"""
Dataset Download & Directory Setup Script.
Prepares directory structure for benchmark datasets (DoTA, CADP, CCD) and local Vietnam Traffic clips.
"""

import os
import argparse
import json


def setup_dataset_structure(base_dir: str = "datasets"):
    """Creates directory layout for datasets."""
    dirs = [
        f"{base_dir}/dota/videos",
        f"{base_dir}/dota/annotations",
        f"{base_dir}/cadp/videos",
        f"{base_dir}/cadp/annotations",
        f"{base_dir}/vn_traffic/images/train",
        f"{base_dir}/vn_traffic/images/val",
        f"{base_dir}/vn_traffic/labels/train",
        f"{base_dir}/vn_traffic/labels/val",
        f"{base_dir}/vn_traffic/videos/accident",
        f"{base_dir}/vn_traffic/videos/normal",
        f"{base_dir}/features/train",
        f"{base_dir}/features/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Successfully initialized dataset directory hierarchy in '{base_dir}'.")

    # Write a sample dataset manifest template
    manifest_file = f"{base_dir}/vn_traffic/manifest.json"
    if not os.path.exists(manifest_file):
        sample_manifest = [
            {
                "video_file": "accident_001.mp4",
                "label": "accident",
                "start_frame": 120,
                "end_frame": 180,
                "accident_type": "motorcycle_collision",
            },
            {
                "video_file": "normal_001.mp4",
                "label": "normal",
                "start_frame": -1,
                "end_frame": -1,
                "accident_type": "none",
            },
        ]
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(sample_manifest, f, indent=2)
        print(f"📄 Created template annotation manifest: {manifest_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize dataset directory and helpers")
    parser.add_argument("--base_dir", type=str, default="datasets", help="Root datasets directory")
    args = parser.parse_args()
    setup_dataset_structure(args.base_dir)
