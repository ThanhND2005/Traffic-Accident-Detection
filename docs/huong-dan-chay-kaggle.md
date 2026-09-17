# 🚀 Hướng Dẫn Chi Tiết Huấn Luyện & Thực Thi Trên Kaggle (kaggle.com)

> **Dự án:** Hệ thống nhận diện tai nạn giao thông từ video  
> **Repository:** [Traffic-Accident-Detection](https://github.com/ThanhND2005/Traffic-Accident-Detection.git)  
> **Template Notebook:** [`notebooks/00_template_colab_kaggle.ipynb`](../notebooks/00_template_colab_kaggle.ipynb)  
> **Thời gian cập nhật:** Tháng 09/2026

---

## 1. Tại Sao Kaggle Là Nền Tảng Huấn Luyện Chính?

Theo kế hoạch triển khai (Mục 2 & 6), Kaggle được chỉ định làm **nguồn GPU chính** cho toàn bộ dự án vì:
- **30 giờ GPU miễn phí / tuần** (cấp cố định vào thứ 7/Chủ nhật hàng tuần).
- Tùy chọn phần cứng mạnh mẽ: **NVIDIA T4 x2 (Dual GPU)** hoặc **NVIDIA P100 (16GB VRAM)**.
- **Tính năng chạy ngầm (Save & Run All)**: Bấm chạy xong có thể tắt máy tính đi ngủ, model vẫn tự train trên server Kaggle tối đa 12 giờ/session mà không lo mất kết nối như Colab.

---

## 2. Bước Chuẩn Bị Tài Khoản (Bắt Buộc Trước Tiên)

> [!IMPORTANT]
> **Xác minh số điện thoại (Phone Verification):**
> Nếu tài khoản Kaggle của bạn chưa xác minh số điện thoại:
> 1. Kaggle sẽ **KHÔNG cho bật Internet** trong notebook (không thể `git clone`, không thể `pip install`).
> 2. Bạn sẽ **KHÔNG được cấp GPU**.
> 
> **Cách làm:** Vào [kaggle.com/settings](https://www.kaggle.com/settings) ➔ cuộn xuống mục **Phone Verification** ➔ nhập số điện thoại để nhận mã OTP xác thực.

---

## 3. Tạo Mới Notebook & Cấu Hình Môi Trường Trên Kaggle

### Bước 3.1: Tạo Notebook mới
1. Truy cập [kaggle.com/code](https://www.kaggle.com/code).
2. Nhấp vào nút **`+ New Notebook`** ở góc trên bên phải.

### Bước 3.2: Cấu hình panel Settings (Thanh công cụ bên phải)
Nhìn sang thanh menu bên phải (biểu tượng mũi tên hoặc bánh răng `Notebook settings`):

| Cài đặt | Giá trị chọn | Giải thích |
|---|---|---|
| **Accelerator** | **`GPU T4 x2`** (hoặc `GPU P100`) | Cấp card đồ họa để train YOLO/LSTM/X3D. |
| **Internet** | **`Internet On`** (Gạt công tắc sang bật) | **Bắt buộc** để tải thư viện và clone code từ GitHub. |
| **Language** | `Python` | Ngôn ngữ thực thi. |
| **Environment** | `Always use latest environment` | Dùng image Docker mới nhất của Kaggle. |
| **Persistence** | `Variables and Files` | Giúp giữ lại biến tạm khi notebook reload. |

---

## 4. Đưa Code Dự Án Lên Kaggle

Có 2 cách nhanh nhất để làm việc với code trên Kaggle:

### Cách A: Upload trực tiếp file Notebook Template (Khuyến nghị)
1. Trong giao diện Kaggle Notebook, chọn menu **`File`** ➔ **`Import Notebook`**.
2. Chọn tab **Computer** ➔ Tải file [`notebooks/00_template_colab_kaggle.ipynb`](../notebooks/00_template_colab_kaggle.ipynb) từ máy tính của bạn lên.
3. Toàn bộ code thiết lập môi trường, kiểm tra GPU, clone repo và hàm quản lý checkpoint sẽ sẵn sàng.

### Cách B: Chạy lệnh clone trực tiếp trong ô Code đầu tiên
Nếu bạn tạo một notebook trắng hoàn toàn, hãy tạo một Cell và chạy đoạn code sau:

```python
# =====================================================================
# CELL 1: SETUP WORKSPACE & CLONE REPOSITORY
# =====================================================================
import os
import sys
from pathlib import Path

# Thư mục làm việc trên Kaggle
WORKSPACE_DIR = Path("/kaggle/working")
REPO_URL = "https://github.com/ThanhND2005/Traffic-Accident-Detection.git"
REPO_NAME = "Traffic-Accident-Detection"
REPO_PATH = WORKSPACE_DIR / REPO_NAME

# Clone repo (hoặc git pull nếu đã tồn tại)
if not REPO_PATH.exists():
    print(f"🔄 Đang clone repo từ {REPO_URL}...")
    !git clone {REPO_URL} {REPO_PATH}
else:
    print(f"🔄 Repo đã có sẵn. Đang pull cập nhật...")
    !cd {REPO_PATH} && git pull origin main

# Chuyển working dir vào thư mục dự án & cấu hình sys.path
os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

print(f"✅ Thư mục hiện tại: {os.getcwd()}")
```

---

## 5. Cài Đặt Dependencies Trên Kaggle

Chạy Cell tiếp theo để cài đặt các thư viện cần thiết:

```python
# =====================================================================
# CELL 2: INSTALL DEPENDENCIES
# =====================================================================
!pip install -q ultralytics opencv-python scipy scikit-learn focal-loss-torch pyyaml

# PyTorchVideo cho nhánh Visual X3D
try:
    import pytorchvideo
    print("PyTorchVideo đã sẵn sàng.")
except ImportError:
    print("Đang cài đặt PyTorchVideo từ Facebook Research...")
    !pip install -q fvcore iopath
    !pip install -q "git+https://github.com/facebookresearch/pytorchvideo.git"

# Kiểm tra GPU nhận diện trong PyTorch
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

---

## 6. Đưa Dữ Liệu (Datasets) & Tài Liệu Từ Google Drive Vào Kaggle

Dữ liệu trên Kaggle không nên tải từng video trực tiếp qua git vì dung lượng lớn. Dưới đây là 3 phương pháp tối ưu nhất để đưa dữ liệu và tài liệu huấn luyện vào Kaggle:

### Cách 1: Tải Trực Tiếp Từ Google Drive Bằng `gdown` (Khuyến Nghị Nhanh Gọn)

`gdown` là thư viện tiêu chuẩn để tải file dung lượng lớn và thư mục trực tiếp từ Google Drive vào môi trường Kaggle mà không cần cấu hình tài khoản phức tạp.

#### Bước A: Cấp quyền chia sẻ công khai trên Google Drive (Bắt buộc)
1. Mở Google Drive, nhấp chuột phải vào file (ví dụ: `dataset_vn_traffic.zip` hoặc `best.pt`) hoặc thư mục cần chia sẻ.
2. Chọn **Share (Chia sẻ)** ➔ Tại mục *General access (Quyền truy cập chung)*, đổi từ "Restricted" thành **"Anyone with the link" (Bất kỳ ai có đường liên kết)** với vai trò **Viewer (Người xem)**.
3. Nhấp **Copy link (Sao chép đường liên kết)**.

#### Bước B: Lấy ID của File hoặc Thư mục (Folder) từ URL
- **Nếu là File:** Link có dạng:  
  `https://drive.google.com/file/d/`**`1ABCxyz987654321_FILE_ID`**`/view?usp=sharing`  
  ➔ Mã ID là chuỗi ký tự nằm giữa `/d/` và `/view`.
- **Nếu là Thư mục (Folder):** Link có dạng:  
  `https://drive.google.com/drive/folders/`**`1XYZabc123456789_FOLDER_ID`**`?usp=sharing`  
  ➔ Mã ID là chuỗi ký tự nằm sau `/folders/`.

#### Bước C: Code tải dữ liệu từ Google Drive (Tự nhận diện File nén `.zip` hoặc File bảng `.csv`/`.json`)
Tạo một Cell trên Kaggle và chạy đoạn mã sau (code tự động nhận diện nếu file là `.zip` thì giải nén, nếu là `.csv`/`.json` thì lưu trực tiếp, không bao giờ bị lỗi `BadZipFile`):

```python
# =====================================================================
# CELL 3A: TẢI DATASET HOẶC FILE ANNOTATION TỪ GOOGLE DRIVE
# =====================================================================
!pip install -q gdown

import os
import zipfile
import gdown
from pathlib import Path

# 1. Điền File ID trên Google Drive
DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"  # <-- Thay ID file của bạn vào đây
TEMP_FILE = "/kaggle/working/downloaded_file"
EXTRACT_DIR = Path("/kaggle/working/Traffic-Accident-Detection/datasets/vn_traffic")

# 2. Tải file từ Google Drive
download_url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
print(f"📥 Đang tải từ Google Drive (ID: {DRIVE_FILE_ID})...")
gdown.download(download_url, TEMP_FILE, quiet=False, fuzzy=True)

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

# 3. Kiểm tra định dạng: Nếu là file nén .zip -> Giải nén; Nếu là file thường (.csv, .json, .txt) -> Giữ nguyên
if zipfile.is_zipfile(TEMP_FILE):
    print(f"📦 Phát hiện file nén .zip! Đang giải nén vào: {EXTRACT_DIR}...")
    with zipfile.ZipFile(TEMP_FILE, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    os.remove(TEMP_FILE)
    print(f"✅ Giải nén hoàn tất! Tổng số files: {len(list(EXTRACT_DIR.rglob('*')))}")
else:
    # Nếu không phải zip (ví dụ Crash_Table.csv, annotations, manifest.json)
    # Đọc thử dòng đầu để xác định tên file phù hợp
    with open(TEMP_FILE, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    
    if "vidname" in first_line or "frame" in first_line:
        dest_file = EXTRACT_DIR / "Crash_Table.csv"
    else:
        dest_file = EXTRACT_DIR / "annotations.csv"

    import shutil
    shutil.move(TEMP_FILE, dest_file)
    print(f"📄 Đây là file bảng/dữ liệu dạng text (không phải file nén .zip).")
    print(f"✅ Đã lưu file thành công tại: {dest_file}")
```

#### Bước D: Tải Checkpoint / Weights từ Google Drive để Resume Training hoặc Test
Khi bạn muốn tải checkpoint đã train từ phiên trước hoặc từ đồng đội lưu trên Google Drive:

```python
# =====================================================================
# CELL 3B: TẢI CHECKPOINT (.PT) TỪ GOOGLE DRIVE ĐỂ RESUME / EVALUATE
# =====================================================================
import gdown
from pathlib import Path

CHECKPOINT_FILE_ID = "YOUR_CHECKPOINT_FILE_ID"  # <-- Thay ID file checkpoint
TARGET_CHECKPOINT_PATH = Path("/kaggle/working/Traffic-Accident-Detection/checkpoints/yolo11s_vn_traffic/last.pt")

TARGET_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
gdown.download(f"https://drive.google.com/uc?id={CHECKPOINT_FILE_ID}", str(TARGET_CHECKPOINT_PATH), quiet=False, fuzzy=True)

print(f"✅ Checkpoint đã sẵn sàng tại: {TARGET_CHECKPOINT_PATH}")
```

#### Bước E: Tải cả thư mục (Folder) từ Google Drive
Nếu dữ liệu/video được để trực tiếp trong một thư mục Google Drive:

```python
# =====================================================================
# CELL 3C: TẢI TOÀN BỘ THƯ MỤC TỪ GOOGLE DRIVE BẰNG GDOWN
# =====================================================================
import gdown

FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID"  # <-- Thay ID folder của bạn
DEST_DIR = "/kaggle/working/Traffic-Accident-Detection/datasets/vn_traffic/videos"

gdown.download_folder(id=FOLDER_ID, output=DEST_DIR, quiet=False, use_cookies=False)
print(f"✅ Đã tải toàn bộ thư mục về: {DEST_DIR}")
```

---

### Cách 2: Tạo Kaggle Dataset Riêng (Tối Ưu Nhất Cho Tập Dữ Liệu Cố Định)
Nếu dataset rất lớn (> 5GB) hoặc bạn phải chạy lại nhiều lần, tải bằng Kaggle Dataset sẽ **nhanh hơn gấp nhiều lần** vì dữ liệu được mount trực tiếp vào hệ thống file chỉ trong 2-3 giây:

1. Tải dataset ảnh/video về máy local hoặc chuẩn bị sẵn file nén zip.
2. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) ➔ Bấm **`+ New Dataset`**.
3. Kéo thả thư mục zip dữ liệu lên ➔ Đặt tên (ví dụ: `vietnam-traffic-accident`).
4. Bấm **Create**.
5. Trong Notebook Kaggle của bạn:
   - Bấm nút **`+ Add Input`** ở góc trên bên phải.
   - Tìm kiếm dataset bạn vừa tạo ➔ Nhấp **`Add`**.
   - Dữ liệu sẽ xuất hiện ở đường dẫn chỉ đọc:  
     `/kaggle/input/vietnam-traffic-accident/...`

---

### Cách 3: Tải từ Roboflow (Cho tập gán nhãn ảnh YOLO)
Nếu dữ liệu được gắn nhãn và quản lý phiên bản trên Roboflow:
```python
# =====================================================================
# CELL 3D: TẢI DATASET TỪ ROBOFLOW
# =====================================================================
!pip install -q roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("workspace-name").project("accident-detection")
version = project.version(1)
dataset = version.download("yolov11", location="/kaggle/working/Traffic-Accident-Detection/datasets/vn_traffic")
```

---

## 7. Quản Lý Checkpoint & Huấn Luyện (YOLO11, LSTM, X3D)

### Nguyên tắc quản lý thư mục trên Kaggle:
- **`/kaggle/input`**: Chỉ đọc (Read-only), chứa dataset bạn đính kèm.
- **`/kaggle/working`**: Ghi được (Read-write), dung lượng tối đa **~20 GB**. Mọi thứ lưu trong thư mục này sẽ hiển thị ở tab **Output** sau khi chạy xong để bạn tải về.

### Ví dụ 1: Fine-tune YOLO11s trên Kaggle (Tuần 5)
```python
from ultralytics import YOLO

# Khởi tạo model từ pretrained weights
model = YOLO("yolo11s.pt")

# Huấn luyện
results = model.train(
    data="configs/data.yaml",
    epochs=80,
    imgsz=640,
    batch=16,             # Nếu dùng T4 x2 hoặc P100 có thể thử batch=32
    patience=15,
    optimizer="AdamW",
    lr0=0.001,
    save_period=10,       # Lưu checkpoint mỗi 10 epoch
    project="/kaggle/working/checkpoints",
    name="yolo11s_vn_traffic",
    resume=False          # Đổi thành True nếu chạy tiếp từ last.pt
)
```

### Ví dụ 2: Resume training nếu session trước bị ngắt
Nếu session trước bị dừng ở epoch 40, trong session mới (sau khi đã tải `last.pt` từ Google Drive theo Mục 6 - Bước D):
```python
# Trỏ đến file last.pt đã tải về từ Google Drive hoặc phiên trước
checkpoint_last = "/kaggle/working/Traffic-Accident-Detection/checkpoints/yolo11s_vn_traffic/last.pt"
model = YOLO(checkpoint_last)
model.train(resume=True)
```

---

## 8. Đồng Bộ & Lưu Trữ Checkpoints / Kết Quả Ngược Lên Google Drive

> [!WARNING]
> Khi phiên làm việc Kaggle kết thúc, toàn bộ nội dung trong `/kaggle/working` sẽ bị xóa nếu bạn không lưu lại. Hãy đồng bộ checkpoint (`best.pt`) và biểu đồ kết quả về Google Drive hoặc máy cá nhân.

### Phương án 1: Đóng gói và tải qua tab Output của Kaggle (Đơn giản & Khuyến nghị nhất)
Thêm một cell ở cuối notebook để nén toàn bộ checkpoints và đồ thị huấn luyện:

```python
# =====================================================================
# CELL CUỐI: ĐÓNG GÓI KẾT QUẢ ĐỂ TẢI VỀ HOẶC UPLOAD GOOGLE DRIVE
# =====================================================================
import shutil
from pathlib import Path

checkpoints_dir = Path("/kaggle/working/checkpoints")
archive_output = "/kaggle/working/traffic_accident_experiment_results"

if checkpoints_dir.exists():
    shutil.make_archive(archive_output, "zip", checkpoints_dir)
    print(f"📦 Đã nén toàn bộ checkpoint & log thành công:")
    print(f"   -> File: {archive_output}.zip")
    print(f"   -> Bạn có thể tải file này tại tab Output bên phải màn hình!")
```

### Phương án 2: Tải lên Google Drive trực tiếp bằng Python Script
Nếu bạn chạy chế độ ngầm **Save & Run All (Commit)** và muốn kết quả tự động gửi về Google Drive:
1. Bạn có thể sử dụng thư viện `pydrive2` hoặc Google Drive API.
2. Hoặc lưu file Service Account JSON vào **Kaggle Secrets** (`Add-ons` ➔ `Secrets`) để notebook tự động xác thực và đẩy file lên thư mục Google Drive của nhóm mà không cần tương tác tay.

---

## 9. Chạy Background Training Không Cần Mở Tab (Commit)

Đây là tính năng giá trị nhất của Kaggle giúp bạn tiết kiệm thời gian:

1. Viết xong code huấn luyện hoàn chỉnh trong notebook.
2. Nhìn lên góc trên bên phải, bấm vào nút **`Save Version`**.
3. Chọn **Version Type**: `Save & Run All (Commit)`.
4. Bấm **Save**.
5. **Bây giờ bạn có thể tắt tab trình duyệt, tắt máy tính hoặc đi làm việc khác!**  
   Hệ thống Kaggle sẽ tự động khởi động một phiên làm việc độc lập chạy từ cell đầu đến cell cuối (tối đa 12 tiếng).
6. Khi hoàn tất:
   - Vào lại trang Notebook của bạn trên Kaggle.
   - Nhấp vào mục **Versions** ➔ Xem log chi tiết.
   - Vào tab **Output** ➔ Tải file kết quả (`traffic_accident_experiment_results.zip` hoặc `best.pt`) về máy hoặc tải lên Google Drive.

---

## 10. Khắc Phục Các Sự Cố Phổ Biến Trên Kaggle

| Lỗi gặp phải | Nguyên nhân | Cách khắc phục |
|---|---|---|
| `BadZipFile: File is not a zip file` | Link Google Drive trỏ tới file văn bản (ví dụ `Crash_Table.csv` ~202KB) hoặc trang HTML chứ không phải file nén `.zip` | 1. Nếu là file bảng nhãn `.csv`: Lưu thẳng vào thư mục dataset mà không cần gọi `zipfile.ZipFile(...)` (xem Bước C Mục 6).<br>2. Nếu muốn tải bộ ảnh/video: Kiểm tra lại xem bạn đã lấy đúng File ID của file `.zip` chứa dataset trên Google Drive chưa. |
| `gdown: Cannot retrieve the public link` / `Permission denied` | File/Folder trên Google Drive chưa được cấp quyền chia sẻ công khai | Mở Google Drive ➔ Nhấp chuột phải vào file ➔ Chọn **Share** ➔ Chuyển quyền sang **"Anyone with the link" (Viewer)**. |
| `Download quota exceeded for this file` | Google Drive chặn tải vì quá nhiều lượt tải trong vòng 24 giờ | 1. Mở file trên web Google Drive ➔ Nhấp chuột phải chọn **Make a copy (Tạo bản sao)** ➔ Lấy ID của bản sao để tải.<br>2. Hoặc upload file thành **Kaggle Dataset** (Cách 2 Mục 6) để dùng lâu dài. |
| File tải về bằng `gdown` chỉ nặng vài KB, không mở được | File quá lớn (>100MB) bị chặn bởi trang cảnh báo quét virus của Google | Đảm bảo thêm tham số `fuzzy=True` trong hàm `gdown.download(url, output, fuzzy=True)` hoặc thêm cờ `--fuzzy` khi chạy dòng lệnh. |
| `FileNotFoundError: /kaggle/working/Traffic-Accident-Detection` | Lệnh `git clone` thất bại do chưa bật Internet hoặc Repo đang ở chế độ Private | 1. Bật **Internet On** ở panel bên phải.<br>2. Đổi repo sang **Public** (Settings trên GitHub) hoặc thêm GitHub Personal Access Token (PAT). |
| `No GPU available` | Hết hạn mức 30 giờ/tuần hoặc Kaggle đang quá tải | Chuyển sang Google Colab tạm thời, hoặc chờ đến Chủ Nhật khi Kaggle reset quota 30h. |
| `Failed to connect to github.com` / `pip error` | Chưa bật Internet | Kiểm tra panel Settings bên phải ➔ Gạt **`Internet On`**. Nếu không thấy mục này, hãy thực hiện **Phone Verification**. |
| `No space left on device` | Thư mục `/kaggle/working` vượt quá 20GB | Xóa bớt file `.zip` tạm thời sau khi giải nén bằng `os.remove()` hoặc `!rm -rf ...`. |
| `CUDA out of memory` | Batch size quá lớn so với 16GB VRAM | Giảm `batch` trong `model.train()` từ `16` xuống `8` hoặc `4`. |
| Session bị timeout sau 12h | Giới hạn tối đa của 1 session Kaggle | Chia training làm 2 chặng (ví dụ: Chặng 1 chạy 40 epoch, Chặng 2 dùng `resume=True` chạy nốt 40 epoch). |

---

## 11. Checklist Nhanh Mỗi Khi Mở Kaggle
- [ ] Số điện thoại đã xác minh trong Settings tài khoản.
- [ ] Accelerator đã chọn **GPU T4 x2** hoặc **GPU P100**.
- [ ] Công tắc **Internet On** đã bật.
- [ ] Link Google Drive (dataset / checkpoint) đã chia sẻ ở chế độ **"Anyone with the link"**.
- [ ] File ID Google Drive đã được điền đúng vào Cell tải dữ liệu.
- [ ] Repository `Traffic-Accident-Detection` đã được clone/pull mới nhất.
- [ ] Đường dẫn `save_period` và output trỏ vào `/kaggle/working/checkpoints`.
- [ ] Dùng **Save & Run All (Commit)** khi huấn luyện dài hạn (> 2 giờ).
