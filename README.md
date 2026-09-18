# 🚦 Traffic Accident Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống phát hiện tai nạn giao thông tự động theo thời gian thực từ video/CCTV, tối ưu hóa cho bối cảnh giao thông hỗn hợp (đặc biệt là mật độ xe máy cao tại Việt Nam).

---

## 📌 Tổng Quan Kiến Trúc

Hệ thống kết hợp mô hình thị giác và phân tích quỹ đạo di chuyển theo kiến trúc **Cascade Multi-Modal Inference**:

```
📹 Video / CCTV Stream
       │
       ▼
[Module 1] YOLO11 (Vehicle & Pedestrian Detection)
       │
       ▼
[Module 2] ByteTrack (Multi-Object Tracking)
       │
       ▼
[Module 3] Motion Feature Extractor (Vận tốc, Gia tốc, Góc lái, IoU, Hội tụ khoảng cách)
       │
       ├────────────────────────────────────────┐
       ▼ (Mọi frame)                            ▼ (Chỉ chạy khi có nghi vấn)
[Module 4a] Motion Branch               [Module 4b] Visual Branch
  - Rule-Based Baseline                   - X3D-S Video Classifier
  - Bidirectional LSTM + Attention          (Cắt ROI 16 frames x 160x160)
       │                                        │
       └───────────────────┬────────────────────┘
                           ▼
                 [Module 5] Fusion Module
                   (Late Fusion / MLP)
                           │
                           ▼
              [Module 6] Post-Processing
                - Temporal Smoothing
                - Temporal NMS & Cooldown
                - Multi-level Alerts (🔴 Alert / 🟡 Warning)
                           │
                           ▼
             🚨 Báo động tai nạn + Bounding Box + Timestamp
```

---

## 📁 Cấu Trúc Thư Mục

```
accident-detection/
├── README.md                     # Tài liệu tổng quan dự án
├── requirements.txt              # Thư viện phụ thuộc
├── setup.py                      # Cấu hình cài đặt package
├── configs/
│   ├── default.yaml              # Cấu hình mặc định toàn bộ pipeline
│   ├── bytetrack_custom.yaml     # Tham số ByteTrack tinh chỉnh cho xe máy VN
│   └── data.yaml                 # Cấu hình dataset định dạng YOLO
├── src/
│   ├── __init__.py
│   ├── pipeline.py               # AccidentDetectionPipeline (bộ điều phối chính)
│   ├── detection/
│   │   ├── __init__.py
│   │   └── yolo_detector.py      # Wrapper cho YOLO11 (Ultralytics)
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── trajectory_manager.py # Quản lý quỹ đạo, lịch sử chuyển động (TrajectoryManager)
│   ├── features/
│   │   ├── __init__.py
│   │   └── motion_features.py    # Trích xuất 15 đặc trưng vận động (MotionFeatureExtractor)
│   ├── classifiers/
│   │   ├── __init__.py
│   │   ├── rule_based.py         # RuleBasedAccidentDetector (Baseline)
│   │   ├── lstm_classifier.py    # TrajectoryAccidentClassifier (Bidirectional LSTM)
│   │   └── video_classifier.py   # X3D Video Classifier wrapper (Visual Branch)
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── fusion.py             # AccidentFusion (Kết hợp điểm 2 nhánh)
│   └── postprocessing/
│       ├── __init__.py
│       └── post_processor.py     # Lọc nhiễu, làm mượt chuỗi thời gian, tạo cảnh báo
├── notebooks/
│   ├── 01_eda.ipynb              # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_detection_tracking.ipynb # Kiểm thử Detection & Tracking
│   ├── 03_baseline_rules.ipynb   # Thử nghiệm & tinh chỉnh Rule-Based Baseline
│   ├── 04_yolo_finetune.ipynb    # Kaggle notebook: Fine-tune YOLO11
│   ├── 05_lstm_training.ipynb    # Kaggle notebook: Train LSTM Trajectory Classifier
│   ├── 06_x3d_finetune.ipynb     # Kaggle notebook: Fine-tune X3D Video Classifier
│   ├── 07_fusion_training.ipynb  # Train & calibrate mô hình Fusion
│   └── 08_demo.ipynb             # Notebook demo end-to-end trên video
├── scripts/
│   ├── download_datasets.py      # Tải hoặc tạo cấu trúc dataset mẫu
│   ├── preprocess_data.py        # Tiền xử lý video thành frames & labels
│   ├── extract_features.py       # Trích xuất đặc trưng trajectory từ video
│   ├── evaluate.py               # Đánh giá độ chính xác (Precision, Recall, F1, FAR/h)
│   └── export_model.py           # Xuất mô hình sang ONNX / TorchScript
├── tests/
│   ├── test_detection.py         # Unit test module Detection
│   ├── test_tracking.py          # Unit test module Tracking & Trajectory
│   ├── test_features.py          # Unit test trích xuất đặc trưng
│   └── test_pipeline.py          # Unit test tích hợp toàn bộ pipeline
├── demo/
│   ├── app.py                    # Giao diện Web tương tác (Gradio)
│   └── sample_videos/            # Thư mục chứa video mẫu để test
└── docs/
    ├── architecture.md           # Thiết kế kiến trúc chi tiết từng module
    ├── training_guide.md         # Hướng dẫn train trên Kaggle & Google Colab
    └── results_report.md         # Báo cáo đánh giá kết quả & so sánh các giai đoạn
```

---

## 🚀 Cài Đặt Nhanh

### 1. Khởi tạo môi trường ảo
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 🎯 Hướng Dẫn Sử Dụng

### 1. Chạy Demo Giao Diện Gradio
Khởi chạy ứng dụng Web trực quan để tải video lên và quan sát cảnh báo theo thời gian thực:
```bash
python demo/app.py
```
Mở trình duyệt tại: `http://127.0.0.1:7860`

### 2. Chạy Pipeline Từ Dòng Lệnh
```bash
python -m src.pipeline --video path/to/traffic_video.mp4 --output runs/output_alert.mp4
```

### 3. Trích Xuất Đặc Trưng Chuyển Động
```bash
python scripts/extract_features.py --video_dir data/videos --output_dir data/features
```

### 4. Đánh Giá Toàn Bộ Hệ Thống
```bash
python scripts/evaluate.py --config configs/default.yaml --test_videos data/test_videos
```

### 5. Chạy Unit Tests
```bash
pytest tests/ -v
```

---

## 📊 Thông Số & Hiệu Năng Mục Tiêu

| Tiêu chí | Mục tiêu | Ghi chú |
|---|---|---|
| **Precision** | ≥ 70% | Tránh báo động giả gây phiền nhiễu |
| **Recall** | ≥ 80% | Bắt được tối đa các vụ va chạm thực tế |
| **F1-Score** | ≥ 0.75 | Đảm bảo cân bằng giữa P và R |
| **FAR/h (False Alarm Rate)** | ≤ 2 lần/giờ | Số cảnh báo sai tối đa trong 1 giờ video |
| **Detection Latency** | ≤ 3 giây | Báo động kịp thời sau va chạm |
| **Processing Speed** | ≥ 15 - 25 FPS | Đạt chuẩn thời gian thực với GPU T4 |

---

## 📄 Bản Quyền
Dự án được phân phối dưới giấy phép MIT License.
