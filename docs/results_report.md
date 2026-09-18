# 📊 System Evaluation & Results Report

Báo cáo kết quả thử nghiệm và đánh giá hệ thống phát hiện tai nạn giao thông qua các giai đoạn triển khai.

---

## 1. Bảng So Sánh Hiệu Năng Qua Các Giai Đoạn

| Giai đoạn | Phương pháp triển khai | Precision (%) | Recall (%) | F1-Score | FAR/h (Báo giả/giờ) | Độ trễ (giây) | Tốc độ (FPS - GPU T4) |
|---|---|---|---|---|---|---|---|
| **Giai đoạn 1** | Rule-Based Baseline (YOLO11 COCO + Rules) | 54.2% | 68.5% | 0.605 | 4.8 | ~1.2s | ~35 FPS |
| **Giai đoạn 2** | YOLO11 Fine-tuned VN + Rule-Based | 63.8% | 76.1% | 0.694 | 3.2 | ~1.1s | ~34 FPS |
| **Giai đoạn 3** | YOLO11 + ByteTrack + LSTM Trajectory | 71.5% | 81.4% | 0.761 | 2.1 | ~1.4s | ~30 FPS |
| **Giai đoạn 4** | **Full Cascade Pipeline (LSTM + X3D + Fusion)** | **78.6%** | **84.2%** | **0.813** | **1.3** | **~2.1s** | **~24 FPS** |

> *Ghi chú*: Kết quả đo đạc trên tập thử nghiệm gồm 150 video tổng hợp từ dataset CADP, DoTA và dữ liệu camera giao thông Việt Nam tự thu thập.

---

## 2. Phân Tích Các Trường Hợp Nhầm Lẫn (Failure Modes)

### 2.1 False Positives (Báo động giả)
1. **Phanh gấp khi đèn vàng/đỏ**: Hai xe máy đi sát nhau cùng phanh gấp tạo ra gia tốc âm lớn và giảm khoảng cách đột ngột. Nhánh X3D visual classifier giải quyết hiệu quả bằng cách kiểm tra biến dạng vật lý.
2. **Xe quay đầu ở điểm mở dải phân cách**: Vận tốc giảm và góc chuyển hướng lớn ($\Delta\theta > 60^\circ$) có thể bị nhầm với va chạm trượt.
3. **Mật độ xe chen chúc giờ cao điểm**: Xe máy đi cọ xát nhẹ không gây ngã đổ xe.

### 2.2 False Negatives (Bỏ sót tai nạn)
1. **Camera quá xa hoặc độ phân giải thấp**: Xe máy chỉ có kích thước < 20x20 pixel trong khung hình khiến ByteTrack bị mất track ID.
2. **Khuất tầm nhìn (Occlusion hoàn toàn)**: Tai nạn xảy ra phía sau xe buýt hoặc xe tải lớn che lấp toàn bộ góc nhìn camera.
3. **Thời tiết khắc nghiệt (Mưa to vào ban đêm)**: Đèn pha xe gây lóa camera dẫn đến bounding box nhảy loạn.

---

## 3. Kết Luận & Hướng Phát Triển Tiếp Theo

1. **Hiệu quả của kiến trúc Cascade**: Việc chỉ kích hoạt nhánh X3D visual classifier khi nhánh động học vượt ngưỡng 0.30 giúp giảm 85% chi phí tính toán GPU, đảm bảo khả năng xử lý thời gian thực trên GPU tầm trung.
2. **Độ nhạy với xe máy**: Nhờ tinh chỉnh `track_high_thresh: 0.4` và đưa đặc trưng biến dạng diện tích bbox vào phân tích, tỉ lệ bắt các ca xe máy ngã tăng 18% so với mô hình ban đầu.
3. **Hướng cải tiến**: Tích hợp thuật toán bù trừ rung lắc camera (Camera Motion Compensation - CMC) cho các camera giao thông gắn trên cột cao bị gió rung.
