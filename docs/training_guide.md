# 📖 Training & Experimentation Guide (Kaggle & Colab)

Tài liệu hướng dẫn chi tiết quy trình huấn luyện các module trong hệ thống nhận diện tai nạn giao thông sử dụng GPU miễn phí từ **Kaggle** (ưu tiên chính) và **Google Colab** (phụ trợ).

---

## 1. Chiến Lược Quản Lý GPU

| Nền tảng | Hạn mức / Tuần | Đặc điểm | Khuyến nghị sử dụng |
|---|---|---|---|
| **Kaggle** | **30 giờ GPU** (T4 x2 hoặc P100) | Ổn định, có background run (commit), không lo đứt mạng | Huấn luyện chính: YOLO11 Fine-tune, LSTM, X3D-S |
| **Google Colab** | Biến động (~10-15h) | Thường bị timeout sau vài giờ nếu không tương tác | Chạy thử nghiệm ngắn, EDA, Preprocessing, Demo |

---

## 2. Hướng Dẫn Train Trên Kaggle

### Bước 1: Tạo Notebook & Bật GPU
1. Truy cập Kaggle Notebooks -> Click **New Notebook**.
2. Tại thanh cài đặt bên phải:
   - **Accelerator**: Chọn `GPU T4 x2` hoặc `GPU P100`.
   - **Internet**: Bật `On` (để tải thư viện và mô hình pretrained).
   - **Persistence**: Chọn `Variables and Files`.

### Bước 2: Clone Repo hoặc Upload Code
```bash
# Trong cell đầu tiên của Kaggle Notebook:
!git clone https://github.com/<your-username>/accident-detection.git
%cd accident-detection
!pip install -r requirements.txt
```

### Bước 3: Fine-tune YOLO11 (Notebook 04)
```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(
    data="configs/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    save_period=10,
    project="checkpoints",
    name="yolo11s_vn_traffic"
)
```

### Bước 4: Train LSTM Trajectory Classifier (Notebook 05)
```python
import torch
import numpy as np
from src.classifiers.lstm_classifier import TrajectoryAccidentClassifier

# Load extracted dataset
data = np.load("datasets/features/train_features.npz")
X, y = data["X"], data["y"]

model = TrajectoryAccidentClassifier(input_dim=15, hidden_dim=64, num_layers=2)
# Huấn luyện với Focal Loss hoặc Weighted BCE
# ... (Xem chi tiết tại notebooks/05_lstm_training.ipynb)
```

---

## 3. Hướng Dẫn Train Trên Google Colab

### Mount Google Drive để lưu checkpoint liên tục
```python
from google.colab import drive
import os
import shutil

drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/accident_detection/checkpoints'
os.makedirs(DRIVE_DIR, exist_ok=True)

# Sau mỗi checkpoint quan trọng:
shutil.copy('checkpoints/best.pt', f'{DRIVE_DIR}/yolo11s_best.pt')
print("✅ Saved checkpoint to Google Drive!")
```

---

## 4. Checklist Trước Khi Bắt Đầu Train
- [ ] Dữ liệu ảnh train/val đã được phân chia theo cấu trúc YOLO.
- [ ] File `configs/data.yaml` đã cập nhật đường dẫn đúng.
- [ ] Kiểm tra kết nối GPU bằng lệnh `torch.cuda.is_available()`.
- [ ] Bật cờ `resume=True` nếu tiếp tục huấn luyện từ checkpoint trước khi phiên làm việc bị ngắt.
