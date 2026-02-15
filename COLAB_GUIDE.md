# 🚀 Hướng Dẫn Chạy Dự Án Trên Google Colab

## 📁 Bước 1: Upload Folder Lên Google Drive

**Cách đơn giản nhất:**
1. Mở Google Drive: https://drive.google.com
2. Kéo thả **nguyên cả folder `DL for disaster`** vào My Drive
3. Đợi upload xong (folder ~500MB với file CSV)

```
MyDrive/
└── DL for disaster/      ← Kéo thả nguyên folder này
    ├── SEA_2024_FINAL_CLEAN.csv
    ├── configs/
    ├── src/
    ├── notebooks/
    └── ...

## 🖥️ Bước 2: Mở Google Colab

1. Vào https://colab.research.google.com
2. File → Open notebook → Google Drive
3. Chọn file `train_colab.ipynb` trong folder `DL for disaster/notebooks/`

## ⚡ Bước 3: Cấu Hình GPU

**QUAN TRỌNG:** Phải làm trước khi chạy!

1. Vào menu **Runtime** → **Change runtime type**
2. Hardware accelerator: Chọn **GPU**
3. GPU type: Chọn **T4** (miễn phí) hoặc **A100/V100** (Colab Pro)
4. Click **Save**

## ▶️ Bước 4: Chạy Notebook

1. **Cell 1**: Mount Google Drive - click "Connect to Google Drive" khi được hỏi
2. **Cell 2**: Install packages
3. **Cell 3**: Check GPU - đảm bảo hiển thị GPU (Tesla T4 hoặc tương tự)
4. Chạy lần lượt các cell còn lại hoặc **Runtime → Run all**

## ⏱️ Thời Gian Ước Tính

| Cấu hình | Thời gian/epoch | Tổng (100 epochs) |
|----------|-----------------|-------------------|
| Local MPS (M1/M2) | ~10-15 phút | ~15-25 giờ |
| Colab T4 | ~2-3 phút | ~3-5 giờ |
| Colab A100 | ~30-60 giây | ~1-2 giờ |

## 💡 Tips Quan Trọng

### 1. Tránh Bị Ngắt Kết Nối
- Colab tự động ngắt sau 90 phút không hoạt động
- Mở tab Colab và không minimize
- Có thể dùng extension "Colab Auto-clicker" để giữ session

### 2. Lưu Model Tự Động
- Model được lưu vào Google Drive (`DL for disaster/models/`)
- Nếu bị ngắt, có thể tiếp tục từ checkpoint

### 3. Tăng Tốc Độ Training
```python
# Trong notebook, tăng batch_size nếu GPU có đủ RAM
config.data.batch_size = 1024  # T4 có 16GB VRAM
```

### 4. Kiểm Tra GPU Usage
```python
# Thêm cell này để monitor GPU
!nvidia-smi
```

## 🔧 Xử Lý Lỗi Thường Gặp

### Lỗi "CUDA out of memory"
```python
# Giảm batch_size
config.data.batch_size = 256
```

### Lỗi "No module named 'src'"
```python
# Kiểm tra đường dẫn
import sys
sys.path.insert(0, '/content/drive/MyDrive/DL for disaster')
```

### Lỗi "File not found"
```python
# Kiểm tra file tồn tại
import os
os.listdir('/content/drive/MyDrive/DL for disaster')
```

## 📊 Sau Khi Training Xong

Các file kết quả sẽ được lưu trong Google Drive:
- `DL for disaster/models/disaster_model_best.pt` - Model tốt nhất
- `DL for disaster/logs/disaster_model_history.json` - Lịch sử training
- `DL for disaster/logs/disaster_model_curves.png` - Biểu đồ training
- `DL for disaster/logs/disaster_model_confusion_matrices.png` - Confusion matrices

## 🎯 Quick Start Commands

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Navigate to project
%cd "/content/drive/MyDrive/DL for disaster"

# Cell 3: Check GPU
!nvidia-smi
```

---
**Lưu ý:** Colab miễn phí giới hạn ~12 giờ/session. Với dữ liệu 1.3M rows và 100 epochs, nên đủ thời gian hoàn thành trên T4 GPU.
