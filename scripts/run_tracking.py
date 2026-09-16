"""
Script chạy YOLO11 + ByteTrack tracking trên video, xuất trajectory ra CSV + JSON.

Đây là script tự động hóa toàn bộ pipeline tuần 2:
  1. Load YOLO11s pretrained
  2. Chạy ByteTrack tracking từng frame
  3. Tích lũy trajectory qua TrajectoryManager
  4. Xuất trajectories.csv + trajectories.json
  5. In thống kê tổng quan

Sử dụng:
    python scripts/run_tracking.py \
        --video path/to/video.mp4 \
        --output-dir results/week2_output \
        --model yolo11s.pt \
        --conf 0.4 \
        --classes 0 1 2 3 5 7

Kaggle usage (thêm --save-video để tạo video annotated):
    python scripts/run_tracking.py \
        --video /kaggle/input/my-dataset/sample.mp4 \
        --output-dir /kaggle/working/week2_output \
        --save-video
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

# Thêm root vào path để import src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.tracking.trajectory_manager import TrajectoryManager, COCO_VEHICLE_CLASSES


# ---------------------------------------------------------------------------
# Tracking pipeline
# ---------------------------------------------------------------------------

def run_tracking(
    video_path: str,
    output_dir: str,
    model_weights: str = "yolo11s.pt",
    tracker_config: str = "configs/bytetrack_custom.yaml",
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    target_classes: list = None,
    max_history: int = 90,
    cleanup_interval: int = 300,
    save_video: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Chạy YOLO11 + ByteTrack trên video, trả về trajectory data.

    Args:
        video_path       : đường dẫn video đầu vào.
        output_dir       : thư mục lưu kết quả.
        model_weights    : file .pt của YOLO11.
        tracker_config   : file yaml của ByteTrack.
        conf_threshold   : ngưỡng confidence detection.
        iou_threshold    : ngưỡng NMS IoU.
        target_classes   : danh sách class ID cần giữ (COCO IDs).
        max_history      : số frame lịch sử tối đa mỗi track.
        cleanup_interval : cleanup track cũ mỗi N frame.
        save_video       : có lưu video annotated không.
        verbose          : in progress không.

    Returns:
        dict với 'summary', 'csv_path', 'json_path', 'video_path'.
    """
    if target_classes is None:
        target_classes = [0, 1, 2, 3, 5, 7]  # COCO vehicle + person

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics chưa được cài. Chạy: pip install ultralytics"
        )

    if verbose:
        print(f"🔄 Loading model: {model_weights}")
    model = YOLO(model_weights)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"🖥️  Device: {device.upper()}")
        if device == "cuda":
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Video info
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if verbose:
        print(f"\n📹 Video: {os.path.basename(video_path)}")
        print(f"   Resolution: {W}×{H}  FPS: {fps:.1f}  Frames: {total_frames}")

    # ------------------------------------------------------------------
    # Writer (nếu save_video)
    # ------------------------------------------------------------------
    writer = None
    annotated_video_path = None
    if save_video:
        annotated_video_path = os.path.join(output_dir, "tracking_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (W, H))

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------
    manager = TrajectoryManager(max_history=max_history)
    t_start = time.time()

    if verbose:
        print(f"\n▶️  Running tracking...")

    # Stream=True để xử lý từng frame, tiết kiệm RAM
    results_gen = model.track(
        source=video_path,
        persist=True,
        tracker=tracker_config,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=target_classes,
        imgsz=640,
        device=device,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for result in results_gen:
        # Trích xuất tracking data
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(np.int32)
            bboxes    = result.boxes.xyxy.cpu().numpy()
            classes   = result.boxes.cls.cpu().numpy().astype(np.int32)
            confs     = result.boxes.conf.cpu().numpy()
            manager.update(frame_idx, track_ids, bboxes, classes, confs)

        # Cleanup định kỳ
        if frame_idx > 0 and frame_idx % cleanup_interval == 0:
            manager.cleanup_old_tracks(frame_idx, max_age=max_history)

        # Lưu frame annotated nếu cần
        if writer is not None and result.plot is not None:
            annotated = result.plot()
            writer.write(annotated)

        frame_idx += 1
        if verbose and frame_idx % 200 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / frame_idx * (total_frames - frame_idx)
            active = len(manager.get_active_tracks())
            print(f"  Frame {frame_idx:5d}/{total_frames}  active={active:3d}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    if writer:
        writer.release()

    elapsed_total = time.time() - t_start
    if verbose:
        print(f"\n✅ Tracking done in {elapsed_total:.1f}s  "
              f"({total_frames / elapsed_total:.1f} FPS effective)")

    # ------------------------------------------------------------------
    # Export kết quả
    # ------------------------------------------------------------------
    csv_path  = os.path.join(output_dir, "trajectories.csv")
    json_path = os.path.join(output_dir, "trajectories.json")

    manager.export_csv(csv_path)
    manager.export_json(json_path)

    summary = manager.get_summary()
    summary_path = os.path.join(output_dir, "tracking_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            **summary,
            "video_path": video_path,
            "total_frames_processed": frame_idx,
            "fps_effective": round(frame_idx / elapsed_total, 2),
            "model": model_weights,
            "conf_threshold": conf_threshold,
        }, f, indent=2)

    if verbose:
        print(f"\n📊 Summary:")
        for k, v in summary.items():
            print(f"   {k}: {v}")
        print(f"\n📁 Output files:")
        print(f"   CSV    : {csv_path}")
        print(f"   JSON   : {json_path}")
        print(f"   Summary: {summary_path}")
        if annotated_video_path:
            print(f"   Video  : {annotated_video_path}")

    return {
        "summary": summary,
        "csv_path": csv_path,
        "json_path": json_path,
        "annotated_video_path": annotated_video_path,
        "summary_path": summary_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO11 + ByteTrack and export trajectories.")
    p.add_argument("--video",       required=True,          help="Input video path")
    p.add_argument("--output-dir",  default="results/week2_output", help="Output directory")
    p.add_argument("--model",       default="yolo11s.pt",   help="YOLO model weights")
    p.add_argument("--tracker",     default="configs/bytetrack_custom.yaml",
                                                            help="ByteTrack yaml config")
    p.add_argument("--conf",        type=float, default=0.4, help="Confidence threshold")
    p.add_argument("--iou",         type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--classes",     type=int, nargs="+",
                   default=[0, 1, 2, 3, 5, 7],             help="COCO class IDs to track")
    p.add_argument("--max-history", type=int, default=90,  help="Max trajectory history frames")
    p.add_argument("--save-video",  action="store_true",   help="Save annotated video")
    p.add_argument("--quiet",       action="store_true",   help="Suppress progress output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracking(
        video_path=args.video,
        output_dir=args.output_dir,
        model_weights=args.model,
        tracker_config=args.tracker,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        target_classes=args.classes,
        max_history=args.max_history,
        save_video=args.save_video,
        verbose=not args.quiet,
    )
