"""
CADP Frame Sequences to MP4 Converter & Ground Truth Parser.
Converts `manual/extracted_frames/<video_name>/*.jpg` into individual MP4 video files
and parses `annotations_CADP.json` for benchmark evaluation in Notebook 03.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
import cv2


def convert_frames_to_video(
    frames_dir: Path,
    output_video_path: Path,
    fps: float = 30.0,
) -> bool:
    """
    Read sorted images from frames_dir and encode them into an MP4 video.
    """
    image_files = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpeg"))
    )
    if not image_files:
        return False

    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        return False

    h, w, _ = first_frame.shape
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    for img_p in image_files:
        frame = cv2.imread(str(img_p))
        if frame is not None:
            writer.write(frame)

    writer.release()
    return True


def process_cadp_dataset(
    cadp_dir: str,
    output_dir: str = "datasets/cadp/videos",
    fps: float = 30.0,
    max_videos: int = None,
):
    cadp_path = Path(cadp_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Locate extracted_frames
    frames_base = None
    possible_frame_dirs = [
        cadp_path / "manual" / "extracted_frames",
        cadp_path / "extracted_frames",
        cadp_path / "manual",
        cadp_path,
    ]
    for cand in possible_frame_dirs:
        if cand.exists():
            subdirs = [d for d in cand.iterdir() if d.is_dir() and any(d.glob("*.jpg"))]
            if subdirs:
                frames_base = cand
                break

    if frames_base is None:
        print(f"❌ Không tìm thấy thư mục chứa chuỗi ảnh frame trong: {cadp_dir}")
        print("   Hãy kiểm tra lại đường dẫn tới thư mục CADP.")
        return

    video_folders = sorted([d for d in frames_base.iterdir() if d.is_dir() and any(d.glob("*.jpg"))])
    if max_videos:
        video_folders = video_folders[:max_videos]

    print(f"📁 Tìm thấy {len(video_folders)} thư mục video clip trong: {frames_base}")
    print(f"🚀 Bắt đầu ghép ảnh thành video MP4 tại: {out_path}\n")

    converted_count = 0
    for idx, vdir in enumerate(video_folders):
        out_vid = out_path / f"{vdir.name}.mp4"
        success = convert_frames_to_video(vdir, out_vid, fps=fps)
        if success:
            converted_count += 1
            if (idx + 1) % 10 == 0 or idx == len(video_folders) - 1:
                print(f"   [{idx + 1:3d}/{len(video_folders)}] ✅ {vdir.name}.mp4")

    print(f"\n🎉 Đã chuyển đổi thành công {converted_count} video MP4 vào: {out_path}")

    # 2. Check and parse annotations_CADP.json if present
    annot_file = None
    possible_annots = [
        cadp_path / "annotations_CADP.json",
        cadp_path / "manual" / "annotations_CADP.json",
    ]
    for a in possible_annots:
        if a.exists():
            annot_file = a
            break

    if annot_file:
        print(f"\n📄 Đang nạp nhãn sự thật chuẩn (Ground Truth): {annot_file.name}...")
        try:
            with open(annot_file, "r", encoding="utf-8") as f:
                annot_data = json.load(f)
            gt_out = out_path.parent / "cadp_ground_truth.json"
            with open(gt_out, "w", encoding="utf-8") as f:
                json.dump(annot_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu file Ground Truth chuẩn hóa cho Notebook 03: {gt_out}")
        except Exception as e:
            print(f"⚠️ Lỗi đọc annotations: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CADP extracted frame folders into MP4 videos")
    parser.add_argument("--cadp_dir", type=str, required=True, help="Path to your CADP folder")
    parser.add_argument("--output_dir", type=str, default="datasets/cadp/videos", help="Target output folder for MP4s")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for output videos")
    parser.add_argument("--max_videos", type=int, default=None, help="Limit number of videos to convert")
    args = parser.parse_args()

    process_cadp_dataset(
        cadp_dir=args.cadp_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        max_videos=args.max_videos,
    )
