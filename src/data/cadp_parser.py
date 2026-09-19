"""
src/data/cadp_parser.py
=============================================================================
MÔ-ĐUN: CADP Dataset Ground Truth Parser & Temporal Labeling Bridge
=============================================================================

MỤC TIÊU KỸ THUẬT:
    1. Đọc và chuẩn hóa file nhãn gốc 'annotation.cadp' (JSON lồng nhau).
    2. Duỗi phẳng (Flatten) dữ liệu thành cấu trúc bảng (DataFrame / CSV) phục vụ
       cho các bài toán: EDA, Group-Stratified Split, và Evaluation.
    3. Cung cấp API gán nhãn thời gian thực (Temporal Alignment): ánh xạ mốc thời gian
       t_sec (hoặc frame_idx / FPS) từ log tracking (YOLO + ByteTrack) sang nhãn
       nhị phân (Binary Ground Truth): is_accident (0 hoặc 1).

ĐẶC TẢ CẤU TRÚC DỮ LIỆU ĐẦU VÀO (annotation.cadp):
    Dictionary dạng key-value:
    {
        "<video_id>.mp4": [
            {
                "Start": 14.5,          # Giây bắt đầu xảy ra tai nạn (float)
                "End": 18.2,            # Giây kết thúc tai nạn (float)
                "bbox": [x1, y1, x2, y2] # (Tùy chọn) Bounding box vị trí va chạm
            }, ...
        ],
        "<video_id_normal>.mp4": []     # Mảng rỗng biểu thị Negative Sample (giao thông bình thường)
    }

ĐẶC TẢ DỮ LIỆU ĐẦU RA (cadp_ground_truth.csv):
    Bảng dữ liệu phẳng gồm các trường:
    - video_filename (str) : Tên file video gốc kèm phần mở rộng (vd: -GpvLzopst8.mp4)
    - video_id (str)       : Khóa chính (Primary Key), đã tách đuôi .mp4 (vd: -GpvLzopst8)
    - has_accident (int)   : 1 nếu video chứa ít nhất 1 vụ tai nạn, 0 nếu là clip bình thường
    - start_sec (float)    : Mốc thời gian bắt đầu tai nạn tính bằng giây (None nếu has_accident=0)
    - end_sec (float)      : Mốc thời gian kết thúc tai nạn tính bằng giây (None nếu has_accident=0)
    - segment_idx (int)    : Thứ tự vụ tai nạn trong clip (hỗ trợ clip có nhiều va chạm liên tiếp)

VÍ DỤ TÍCH HỢP VỚI TRACKING PIPELINE (Tuần 3 & Tuần 5):
    >>> from src.data.cadp_parser import CADPParser
    >>> parser = CADPParser("data/annotation.cadp")
    >>> # Giả sử df_tracking có cột 'video_id' và 't_sec' (hoặc frame_idx / fps):
    >>> df_tracking['is_accident'] = df_tracking.apply(
    ...     lambda r: parser.is_accident_frame(r['video_id'], r['t_sec']), axis=1
    ... )
=============================================================================
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional


class CADPParser:
    """
    Parser chuẩn hóa nhãn annotation.cadp cho CADP (CCTV-based Accident Detection Dataset).
    """

    def __init__(self, annotation_path: str):
        """
        Khởi tạo parser, nạp dữ liệu thô và tự động duỗi phẳng thành DataFrame.

        Args:
            annotation_path (str): Đường dẫn vật lý đến file annotation.cadp
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Không tìm thấy file annotation tại: {annotation_path}")
        
        self.annotation_path = annotation_path
        self.raw_data = self._load_annotation()
        self.ground_truth_df = self._parse_to_dataframe()

    def _load_annotation(self) -> Dict[str, Any]:
        """Đọc và giải mã an toàn file JSON annotation.cadp."""
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _parse_to_dataframe(self) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu JSON thô thành bảng phẳng (Flattened DataFrame).
        Cơ chế phòng vệ: Hỗ trợ linh hoạt cả key viết hoa (Start/End) lẫn viết thường (start/end).
        Bảo toàn mẫu âm (Negative samples) với các video có mảng segment rỗng [].
        """
        records = []

        for video_filename, segments in self.raw_data.items():
            # Chuẩn hóa Primary Key: loại bỏ đuôi mở rộng file (.mp4, .avi, ...)
            video_id = os.path.splitext(video_filename)[0]

            # 1. Trường hợp Mẫu âm (Negative Sample): Không có va chạm trong video
            if not segments:
                records.append({
                    "video_filename": video_filename,
                    "video_id": video_id,
                    "has_accident": 0,
                    "start_sec": None,
                    "end_sec": None,
                    "segment_idx": None
                })
                continue

            # 2. Trường hợp Mẫu dương (Positive Sample): Chứa 1 hoặc nhiều đoạn tai nạn
            for seg_idx, seg in enumerate(segments):
                start = seg.get("Start", seg.get("start", None))
                end = seg.get("End", seg.get("end", None))

                records.append({
                    "video_filename": video_filename,
                    "video_id": video_id,
                    "has_accident": 1,
                    "start_sec": float(start) if start is not None else None,
                    "end_sec": float(end) if end is not None else None,
                    "segment_idx": seg_idx
                })

        df = pd.DataFrame(records)
        return df

    def export_manifest(self, output_csv_path: str):
        """
        Xuất bảng Ground Truth Manifest ra định dạng CSV để dùng chung cho cả team.

        Args:
            output_csv_path (str): Đường dẫn lưu file CSV đầu ra.
        """
        parent_dir = os.path.dirname(output_csv_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.ground_truth_df.to_csv(output_csv_path, index=False)
        print(f"✅ Đã xuất Ground Truth CADP ({len(self.ground_truth_df)} dòng) ra: {output_csv_path}")

    def is_accident_frame(self, video_id: str, t_sec: float) -> bool:
        """
        Truy vấn kiểm tra tại thời điểm t_sec của video_id có xảy ra va chạm hay không.

        Thuật toán:
            - Lọc các khoảng va chạm [start_sec, end_sec] tương ứng với video_id.
            - Kiểm tra điều kiện bao hàm: start_sec <= t_sec <= end_sec.

        Args:
            video_id (str): Mã định danh video (Primary Key).
            t_sec (float): Thời điểm khung hình tính bằng giây (frame_idx / FPS).

        Returns:
            bool: True nếu khung hình thuộc khoảng tai nạn, ngược lại False.
        """
        sub = self.ground_truth_df[
            (self.ground_truth_df["video_id"] == video_id) & 
            (self.ground_truth_df["has_accident"] == 1)
        ]

        if sub.empty:
            return False

        hit = sub[(sub["start_sec"] <= t_sec) & (t_sec <= sub["end_sec"])]
        return not hit.empty


# ── Test nhanh độc lập (Standalone Execution) ─────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parser chuẩn hóa CADP Ground Truth")
    parser.add_argument("--anno_path", type=str, default="data/annotation.cadp", help="Đường dẫn tới annotation.cadp")
    parser.add_argument("--output_csv", type=str, default="data/cadp_ground_truth.csv", help="Đường dẫn file CSV đầu ra")
    args = parser.parse_args()

    if os.path.exists(args.anno_path):
        cadp = CADPParser(args.anno_path)
        cadp.export_manifest(args.output_csv)
        
        # Test kiểm thử tính nhất quán
        sample_vids = cadp.ground_truth_df[cadp.ground_truth_df["has_accident"] == 1]
        if not sample_vids.empty:
            sample_row = sample_vids.iloc[0]
            vid = sample_row["video_id"]
            mid_time = (sample_row["start_sec"] + sample_row["end_sec"]) / 2
            is_hit = cadp.is_accident_frame(vid, mid_time)
            print(f"\n🔍 [Sanity Check] Video='{vid}' tại t={mid_time:.2f}s -> is_accident={is_hit}")
    else:
        print(f"⚠️ Không tìm thấy file tại '{args.anno_path}'. Vui lòng chỉ định bằng flag: --anno_path <path>")