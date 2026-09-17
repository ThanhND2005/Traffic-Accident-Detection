"""
Script trực quan hóa trajectory trên video.

Dùng sau khi chạy tracking (notebook 02), để tạo video demo với:
  - Bounding box + track ID + class label
  - Đường trajectory màu sắc cho từng xe (mờ dần về quá khứ)
  - Thống kê real-time: số track, FPS

Sử dụng:
    python scripts/visualize_trajectory.py \
        --video  path/to/input.mp4 \
        --tracks path/to/trajectories.json \
        --output path/to/output_viz.mp4 \
        --tail   30
"""

import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
COCO_VEHICLE_CLASSES = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorcycle", 5: "bus", 7: "truck",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectories(json_path: str) -> dict:
    """Load file JSON xuất từ TrajectoryManager.export_json()."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tổ chức lại: frame_id → list of track states
    frame_map = defaultdict(list)
    for tid_str, states in data["tracks"].items():
        tid = int(tid_str)
        for s in states:
            frame_map[s["frame_id"]].append({**s, "track_id": tid})

    # Lịch sử center theo track_id
    history_map = {}
    for tid_str, states in data["tracks"].items():
        tid = int(tid_str)
        history_map[tid] = [(s["frame_id"], s["center"]) for s in states]

    return frame_map, history_map


def make_color(track_id: int) -> tuple:
    """Tạo màu deterministic từ track_id (không random mỗi lần chạy)."""
    rng = random.Random(track_id * 2654435761)
    r = rng.randint(80, 255)
    g = rng.randint(80, 255)
    b = rng.randint(80, 255)
    return (b, g, r)  # BGR cho OpenCV


def draw_dashed_bbox(frame, x1, y1, x2, y2, color, thickness=2, dash_len=10):
    """Vẽ bounding box với đường nét đứt (dashed)."""
    pts = [(x1, y1, x2, y1), (x2, y1, x2, y2),
           (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for ax, ay, bx, by in pts:
        length = int(np.hypot(bx - ax, by - ay))
        if length == 0:
            continue
        dx = (bx - ax) / length
        dy = (by - ay) / length
        seg = 0
        drawing = True
        while seg < length:
            if drawing:
                ex = int(ax + dx * min(seg + dash_len, length))
                ey = int(ay + dy * min(seg + dash_len, length))
                cv2.line(frame, (int(ax + dx * seg), int(ay + dy * seg)),
                         (ex, ey), color, thickness)
            seg += dash_len
            drawing = not drawing


def draw_label(frame, text, x, y, color, font_scale=0.45, thickness=1):
    """Vẽ label có nền mờ."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    pad = 3
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize(video_path: str, json_path: str, output_path: str, tail_frames: int = 30):
    """
    Tạo video đầu ra với bbox, track ID, và đường trajectory.

    Args:
        video_path   : đường dẫn video đầu vào.
        json_path    : file JSON từ TrajectoryManager.export_json().
        output_path  : đường dẫn video đầu ra.
        tail_frames  : số frame quá khứ để vẽ đuôi trajectory.
    """
    frame_map, history_map = load_trajectories(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    print(f"📹 Input : {video_path}  ({W}×{H} @ {fps:.1f}fps, {total} frames)")
    print(f"📤 Output: {output_path}")
    print(f"🔍 JSON  : {json_path}")
    print(f"🐾 Tail  : {tail_frames} frames")
    print("Processing...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        states_this_frame = frame_map.get(frame_idx, [])

        # 1. Vẽ đường trajectory (đuôi mờ dần)
        for tid, history in history_map.items():
            # Lọc các điểm trong khoảng [frame_idx - tail, frame_idx]
            tail = [
                (f, c) for f, c in history
                if frame_idx - tail_frames <= f <= frame_idx
            ]
            if len(tail) >= 2:
                color = make_color(tid)
                for i in range(1, len(tail)):
                    alpha = i / len(tail)          # mờ dần về quá khứ
                    c = tuple(int(v * alpha) for v in color)
                    p1 = tuple(map(int, tail[i - 1][1]))
                    p2 = tuple(map(int, tail[i][1]))
                    cv2.line(frame, p1, p2, c, 2, cv2.LINE_AA)
                # Chấm tròn tại vị trí hiện tại
                if tail:
                    cur = tuple(map(int, tail[-1][1]))
                    cv2.circle(frame, cur, 4, color, -1)

        # 2. Vẽ bounding box + label cho frame hiện tại
        for s in states_this_frame:
            tid = s["track_id"]
            bbox = s["bbox"]  # [x1, y1, x2, y2]
            cls_id = s["class_id"]
            conf = s["confidence"]
            color = make_color(tid)

            x1, y1, x2, y2 = map(int, bbox)
            draw_dashed_bbox(frame, x1, y1, x2, y2, color, thickness=2)

            cls_name = COCO_VEHICLE_CLASSES.get(cls_id, str(cls_id))
            label = f"{cls_name} #{tid} {conf:.2f}"
            draw_label(frame, label, x1, max(y1 - 5, 10), color)

        # 3. Overlay thông tin
        active_count = len(states_this_frame)
        info_lines = [
            f"Frame: {frame_idx}/{total}",
            f"Active tracks: {active_count}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  [{frame_idx:5d}/{total}] active={active_count}")

    cap.release()
    writer.release()
    print(f"\n✅ Done! Video saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tracking trajectories on video.")
    parser.add_argument("--video",  required=True, help="Path to input video file")
    parser.add_argument("--tracks", required=True, help="Path to trajectories.json")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--tail",   type=int, default=30,
                        help="Number of past frames to draw as trajectory tail (default: 30)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(args.video, args.tracks, args.output, args.tail)
