# Depth Anything V2 - Dual Camera Depth System for Jetson Nano

Hệ thống tạo depth map realtime từ 2 camera USB OV9732 trên Jetson Nano,
hiển thị kết quả qua giao diện web (headless).

## Yêu cầu hệ thống

- Jetson Nano (JetPack 4.6+)
- Ubuntu 20.04
- Python 3.8
- OpenCV 4.8 with CUDA
- 2x Camera USB OV9732
- PyTorch 1.13 for Jetson (NVIDIA wheel)

## Cài đặt

### 1. Cài PyTorch cho Jetson Nano

```bash
# Tải PyTorch wheel từ NVIDIA
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.13.0-cp38-cp38-linux_aarch64.whl
pip3 install torch-1.13.0-cp38-cp38-linux_aarch64.whl

# Cài torchvision
git clone --branch v0.14.0 https://github.com/pytorch/vision torchvision
cd torchvision
python3 setup.py install
cd ..
```

### 2. Cài dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Tải model Depth Anything V2

```bash
python3 download_model.py
```

## Chạy ứng dụng

```bash
python3 main.py
```

Sau đó mở trình duyệt trên máy tính cùng mạng LAN, truy cập:

```
http://<jetson-ip>:5000
```

## Cấu hình

Chỉnh sửa `config.py` để thay đổi:
- Camera index (mặc định: 0 và 1)
- Độ phân giải camera
- Model size (vits/vitb)
- Web server port
- FPS target

## Cấu trúc project

```
├── main.py                 # Entry point
├── config.py               # Cấu hình hệ thống
├── camera_manager.py       # Quản lý 2 camera USB
├── depth_estimator.py      # Depth Anything V2 inference
├── web_server.py           # Flask web server streaming
├── download_model.py       # Script tải model
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Giao diện web
└── static/
    └── style.css           # CSS
```
