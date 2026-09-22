"""
Trajectory Management Module — Tuần 2.
Lưu trữ và truy vấn quỹ đạo di chuyển của tất cả object được track.
Cung cấp: center, velocity, acceleration, IoU, khoảng cách, export CSV/JSON.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import json
import csv
import os


# ---------------------------------------------------------------------------
# Hằng số class COCO dùng trong tuần 2 (pretrained COCO IDs)
# ---------------------------------------------------------------------------
COCO_VEHICLE_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class TrajectoryManager:
    """
    Quản lý quỹ đạo di chuyển của tất cả object được track.

    Mỗi track lưu chuỗi trạng thái theo frame:
        frame_id, bbox [x1,y1,x2,y2], center [cx,cy], class_id, conf, area

    Cung cấp:
        - update()                 : cập nhật từ kết quả tracking mỗi frame
        - get_trajectory()         : lấy chuỗi tâm bbox theo thời gian
        - get_velocity()           : vận tốc (pixel/frame)
        - get_speed()              : tốc độ (magnitude)
        - get_acceleration()       : gia tốc
        - get_active_tracks()      : tracks đang active
        - get_interacting_pairs()  : cặp track gần nhau (nguy cơ va chạm)
        - cleanup_old_tracks()     : xóa track quá cũ
        - export_csv()             : xuất ra CSV
        - export_json()            : xuất ra JSON
        - get_summary()            : thống kê tổng quan
    """

    def __init__(self, max_history: int = 90):
        """
        Args:
            max_history: Số frame tối đa lưu cho mỗi track.
                         90 frame ≈ 3 giây ở 30 fps.
        """
        self.max_history = max_history
        # track_id → deque of state dicts
        self.tracks: Dict[int, deque] = {}
        # track_id → frame_id lần cuối thấy
        self.last_seen: Dict[int, int] = {}
        # frame hiện tại (để get_active_tracks không cần truyền tham số)
        self.current_frame: int = 0

    # ------------------------------------------------------------------
    # CORE: Cập nhật + truy vấn
    # ------------------------------------------------------------------

    def update(
        self,
        frame_id: int,
        track_ids: np.ndarray,
        bboxes: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
    ):
        """
        Cập nhật trajectory từ kết quả tracking mỗi frame.

        Args:
            frame_id  : index frame hiện tại (int).
            track_ids : (N,) array — ID của từng track.
            bboxes    : (N, 4) array — [x1, y1, x2, y2].
            classes   : (N,) array — class ID.
            confs     : (N,) array — confidence score.
        """
        self.current_frame = frame_id

        for i in range(len(track_ids)):
            tid = int(track_ids[i])
            if tid < 0:
                continue  # track chưa khởi tạo

            bbox = np.array(bboxes[i], dtype=np.float32)
            cls_id = int(classes[i])
            conf = float(confs[i])
            cx = float((bbox[0] + bbox[2]) / 2.0)
            cy = float((bbox[1] + bbox[3]) / 2.0)
            w = float(bbox[2] - bbox[0])
            h = float(bbox[3] - bbox[1])

            if tid not in self.tracks:
                self.tracks[tid] = deque(maxlen=self.max_history)

            self.tracks[tid].append({
                "frame_id": frame_id,
                "bbox": bbox,                              # (4,) float32
                "center": np.array([cx, cy], dtype=np.float32),
                "width": w,
                "height": h,
                "area": w * h,
                "class_id": cls_id,
                "conf": conf,
            })
            self.last_seen[tid] = frame_id

    # ------------------------------------------------------------------
    # Truy vấn trajectory
    # ------------------------------------------------------------------

    def get_trajectory(self, track_id: int) -> np.ndarray:
        """
        Lấy chuỗi tâm bbox theo thời gian.

        Returns:
            (T, 2) array — [cx, cy] mỗi frame. (0,2) nếu không tìm thấy.
        """
        if track_id not in self.tracks or len(self.tracks[track_id]) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(
            [s["center"] for s in self.tracks[track_id]], dtype=np.float32
        )

    def get_bboxes(self, track_id: int) -> np.ndarray:
        """
        Lấy chuỗi bounding box theo thời gian.

        Returns:
            (T, 4) array — [x1, y1, x2, y2].
        """
        if track_id not in self.tracks or len(self.tracks[track_id]) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(
            [s["bbox"] for s in self.tracks[track_id]], dtype=np.float32
        )

    def get_frame_ids(self, track_id: int) -> np.ndarray:
        """Trả về array các frame_id trong lịch sử track."""
        if track_id not in self.tracks:
            return np.array([], dtype=np.int32)
        return np.array(
            [s["frame_id"] for s in self.tracks[track_id]], dtype=np.int32
        )

    def get_last_state(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Trả về trạng thái mới nhất của track."""
        if track_id in self.tracks and len(self.tracks[track_id]) > 0:
            return self.tracks[track_id][-1]
        return None

    # ------------------------------------------------------------------
    # Đặc trưng chuyển động
    # ------------------------------------------------------------------

    def get_velocity(self, track_id: int) -> Optional[np.ndarray]:
        """
        Tính vận tốc: v(t) = center(t) - center(t-1).

        Returns:
            (T-1, 2) array — [vx, vy] hoặc None nếu quá ngắn.
        """
        traj = self.get_trajectory(track_id)
        if len(traj) < 2:
            return None
        return np.diff(traj, axis=0)  # (T-1, 2)

    def get_speed(self, track_id: int) -> Optional[np.ndarray]:
        """
        Tốc độ (magnitude of velocity) theo từng frame.

        Returns:
            (T-1,) array — pixel/frame.
        """
        vel = self.get_velocity(track_id)
        if vel is None:
            return None
        return np.linalg.norm(vel, axis=1)

    def get_acceleration(self, track_id: int) -> Optional[np.ndarray]:
        """
        Gia tốc: a(t) = v(t) - v(t-1).

        Returns:
            (T-2, 2) array — [ax, ay] hoặc None nếu quá ngắn.
        """
        vel = self.get_velocity(track_id)
        if vel is None or len(vel) < 2:
            return None
        return np.diff(vel, axis=0)

    def get_direction_changes(self, track_id: int) -> Optional[np.ndarray]:
        """
        Góc lệch hướng Δθ(t) = θ(t) - θ(t-1), đơn vị độ.
        Đột biến lớn (>45°) → xe bị văng, xoay đột ngột.

        Returns:
            (T-2,) array — độ lệch góc.
        """
        vel = self.get_velocity(track_id)
        if vel is None or len(vel) < 2:
            return None
        angles = np.arctan2(vel[:, 1], vel[:, 0])          # (T-1,) radian
        delta = np.diff(np.degrees(angles))                 # (T-2,) degrees
        # Chuẩn hóa về [-180, 180]
        delta = (delta + 180) % 360 - 180
        return delta

    def get_area_ratio(self, track_id: int) -> Optional[np.ndarray]:
        """
        Tỉ lệ thay đổi diện tích bbox: area(t) / area(t-k).
        Thay đổi đột ngột → xe bị biến dạng / lật.

        Returns:
            (T-1,) array — tỉ lệ.
        """
        if track_id not in self.tracks:
            return None
        areas = np.array(
            [s["area"] for s in self.tracks[track_id]], dtype=np.float32
        )
        if len(areas) < 2:
            return None
        # Tránh chia 0
        prev = areas[:-1]
        prev = np.where(prev == 0, 1e-6, prev)
        return areas[1:] / prev

    @staticmethod
    def compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
        """
        Tính IoU giữa 2 bounding box [x1, y1, x2, y2].
        IoU > 0.15 giữa 2 xe → 2 xe bắt đầu chồng lên nhau (va chạm).
        """
        xa1 = max(bbox_a[0], bbox_b[0])
        ya1 = max(bbox_a[1], bbox_b[1])
        xa2 = min(bbox_a[2], bbox_b[2])
        ya2 = min(bbox_a[3], bbox_b[3])

        inter_w = max(0.0, xa2 - xa1)
        inter_h = max(0.0, ya2 - ya1)
        inter_area = inter_w * inter_h

        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    # ------------------------------------------------------------------
    # Quản lý tracks
    # ------------------------------------------------------------------

    def get_active_tracks(
        self,
        current_frame: Optional[int] = None,
        min_length: int = 10,
        max_age: Optional[int] = 30,
    ) -> List[int]:
        """
        Lấy danh sách track ID đang active.

        Args:
            current_frame : frame hiện tại (mặc định dùng self.current_frame).
            min_length    : số frame tối thiểu có trong lịch sử.
            max_age       : track phải được thấy trong 'max_age' frame gần nhất.
                            Nếu None, không lọc theo max_age (lấy tất cả track >= min_length).

        Returns:
            List[int] — danh sách track ID hợp lệ.
        """
        if current_frame is None:
            current_frame = self.current_frame

        active = []
        for tid, history in self.tracks.items():
            if len(history) < min_length:
                continue
            if max_age is not None and (current_frame - self.last_seen[tid]) > max_age:
                continue
            active.append(tid)
        return active

    def get_all_tracks(self, min_length: int = 1) -> List[int]:
        """
        Lấy danh sách tất cả track ID có độ dài >= min_length đã từng xuất hiện.
        """
        return [tid for tid, h in self.tracks.items() if len(h) >= min_length]

    def get_interacting_pairs(
        self,
        current_frame: Optional[int] = None,
        distance_threshold: float = 120.0,
        min_length: int = 5,
    ) -> List[Tuple[int, int]]:
        """
        Tìm các cặp track gần nhau (khoảng cách tâm < distance_threshold pixel).
        Dùng để phân tích nguy cơ va chạm.

        Returns:
            List[(tid_a, tid_b)] — danh sách cặp track gần nhau.
        """
        if current_frame is None:
            current_frame = self.current_frame

        active = self.get_active_tracks(current_frame, min_length=min_length, max_age=2)
        pairs = []
        n = len(active)
        for i in range(n):
            c_a = self.tracks[active[i]][-1]["center"]
            for j in range(i + 1, n):
                c_b = self.tracks[active[j]][-1]["center"]
                if np.linalg.norm(c_a - c_b) <= distance_threshold:
                    pairs.append((active[i], active[j]))
        return pairs

    def cleanup_old_tracks(self, current_frame: Optional[int] = None, max_age: int = 60):
        """
        Xóa các track không được thấy trong max_age frame để tiết kiệm RAM.

        Args:
            current_frame : frame hiện tại.
            max_age       : số frame không thấy → xóa.
        """
        if current_frame is None:
            current_frame = self.current_frame

        to_delete = [
            tid for tid, last_f in self.last_seen.items()
            if (current_frame - last_f) > max_age
        ]
        for tid in to_delete:
            self.tracks.pop(tid, None)
            self.last_seen.pop(tid, None)

        if to_delete:
            print(f"  🗑️  Cleaned {len(to_delete)} old tracks at frame {current_frame}")

    # ------------------------------------------------------------------
    # Smoothing (giảm nhiễu trước khi tính đặc trưng)
    # ------------------------------------------------------------------

    def smooth_trajectory(self, track_id: int, window: int = 5) -> Optional[np.ndarray]:
        """
        Làm mượt trajectory bằng moving average.
        Giảm nhiễu do detection jitter trước khi tính velocity / acceleration.

        Args:
            window : kích thước cửa sổ averaging.

        Returns:
            (T, 2) array — trajectory đã làm mượt.
        """
        traj = self.get_trajectory(track_id)
        if len(traj) < window:
            return traj

        smoothed = np.zeros_like(traj)
        half = window // 2
        for i in range(len(traj)):
            lo = max(0, i - half)
            hi = min(len(traj), i + half + 1)
            smoothed[i] = traj[lo:hi].mean(axis=0)
        return smoothed

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, output_path: str):
        """
        Xuất toàn bộ trajectory data ra file CSV.
        Phù hợp để mở bằng Excel / Google Sheets / pandas.

        Columns: track_id, frame_id, center_x, center_y,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 width, height, area, class_id, confidence
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fieldnames = [
            "track_id", "frame_id",
            "center_x", "center_y",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "width", "height", "area",
            "class_id", "class_name", "confidence",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tid, history in sorted(self.tracks.items()):
                for s in history:
                    bbox = s["bbox"]
                    writer.writerow({
                        "track_id": tid,
                        "frame_id": s["frame_id"],
                        "center_x": round(s["center"][0], 2),
                        "center_y": round(s["center"][1], 2),
                        "bbox_x1": round(float(bbox[0]), 2),
                        "bbox_y1": round(float(bbox[1]), 2),
                        "bbox_x2": round(float(bbox[2]), 2),
                        "bbox_y2": round(float(bbox[3]), 2),
                        "width": round(s["width"], 2),
                        "height": round(s["height"], 2),
                        "area": round(s["area"], 2),
                        "class_id": s["class_id"],
                        "class_name": COCO_VEHICLE_CLASSES.get(s["class_id"], str(s["class_id"])),
                        "confidence": round(s["conf"], 4),
                    })

        print(f"✅ CSV saved: {output_path}")

    def export_json(self, output_path: str):
        """
        Xuất toàn bộ trajectory data ra file JSON.
        Giữ cấu trúc đầy đủ, phù hợp cho các module downstream (tuần 3+).

        Schema:
        {
          "meta": {...},
          "tracks": {
            "<track_id>": [
              {"frame_id": int, "center": [cx, cy], "bbox": [x1,y1,x2,y2],
               "class_id": int, "class_name": str, "confidence": float,
               "width": float, "height": float, "area": float},
              ...
            ],
            ...
          }
        }
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        tracks_serialized = {}
        for tid, history in sorted(self.tracks.items()):
            track_list = []
            for s in history:
                bbox = s["bbox"]
                track_list.append({
                    "frame_id": s["frame_id"],
                    "center": [round(float(s["center"][0]), 2),
                               round(float(s["center"][1]), 2)],
                    "bbox": [round(float(bbox[0]), 2), round(float(bbox[1]), 2),
                             round(float(bbox[2]), 2), round(float(bbox[3]), 2)],
                    "class_id": s["class_id"],
                    "class_name": COCO_VEHICLE_CLASSES.get(s["class_id"], str(s["class_id"])),
                    "confidence": round(s["conf"], 4),
                    "width": round(s["width"], 2),
                    "height": round(s["height"], 2),
                    "area": round(s["area"], 2),
                })
            tracks_serialized[str(tid)] = track_list

        output = {
            "meta": {
                "total_tracks": len(self.tracks),
                "max_history": self.max_history,
                "last_frame": self.current_frame,
            },
            "tracks": tracks_serialized,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON saved: {output_path}")

    # ------------------------------------------------------------------
    # Thống kê
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """
        Thống kê tổng quan về các track đang lưu.

        Returns:
            dict với các key: total_tracks, avg_length, max_length,
                              active_tracks, current_frame, class_distribution.
        """
        if not self.tracks:
            return {"total_tracks": 0, "current_frame": self.current_frame}

        lengths = [len(h) for h in self.tracks.values()]
        active = self.get_active_tracks()

        # Phân bố class
        class_count: Dict[str, int] = {}
        for history in self.tracks.values():
            if history:
                last = history[-1]
                cname = COCO_VEHICLE_CLASSES.get(last["class_id"], str(last["class_id"]))
                class_count[cname] = class_count.get(cname, 0) + 1

        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(active),
            "avg_track_length": round(float(np.mean(lengths)), 1),
            "max_track_length": int(np.max(lengths)),
            "min_track_length": int(np.min(lengths)),
            "current_frame": self.current_frame,
            "class_distribution": class_count,
        }

    def __repr__(self) -> str:
        s = self.get_summary()
        return (
            f"TrajectoryManager("
            f"tracks={s['total_tracks']}, "
            f"active={s.get('active_tracks', 0)}, "
            f"frame={s['current_frame']})"
        )