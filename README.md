# 🚗 Autonomous Road Anomaly Detection & Geo-Tagging System

**Edge-Based Multi-Agent Infrastructure Monitoring System**

## 📌 System Overview

- **Hardware**: NVIDIA Jetson Nano/Orin Nano
- **Dataset**: RDD2022 India (7,706 images)
- **Framework**: Ultralytics YOLOv8
- **Models**: YOLOv8-Small / YOLOv8-Medium (configurable)
- **Deployment**: TensorRT optimization for Jetson

## 🎯 Project Objectives

1. Real-time road anomaly detection
2. Classification into 4 damage categories
3. GPS coordinate capture (lat/long)
4. Automated reporting to municipal authorities
5. Edge deployment on Jetson devices

## 🏗️ System Architecture

```
Camera → Detection Agent → GPS Tagging Agent → Reporting Agent → Database
```

### Components:
- **Detection Agent**: YOLOv8-based anomaly detector
- **GPS Tagging Agent**: Geo-location capture system
- **Reporting Agent**: Automated notification system (email/webhook)

## 📊 Dataset Classes

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | Pothole | Circular/oval depressions |
| 1 | Alligator Crack | Interconnected cracks |
| 2 | Longitudinal Crack | Linear cracks along road |
| 3 | Other Damage | Miscellaneous damage |

## 📁 Project Structure

```
Project/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── 🌐 Applications
│   ├── streamlit_app.py           # Web GUI (recommended)
│   ├── realtime_detection.py      # Real-time detection with GUI
│   ├── detect_demo.py             # Simple CLI detection
│   ├── batch_test.py              # Batch processing
│   └── setup_verify.py            # Setup verification
│
├── 🤖 jetson/                      # Jetson deployment
│   ├── inference.py               # Optimized inference
│   ├── tensorrt_inference.py      # TensorRT engine
│   ├── memory_monitor.py          # Memory monitoring
│   ├── main_system.py             # Multi-agent system
│   ├── gps_agent.py               # GPS tagging
│   ├── reporting_agent.py         # Automated reporting
│   ├── convert_tensorrt_fp16.sh   # TensorRT FP16 conversion
│   ├── convert_tensorrt_int8.sh   # TensorRT INT8 conversion
│   └── benchmark_engines.sh       # Performance benchmarking
│
├── ⚙️ config/                      # Configuration files
│   └── deployment_config.json     # Deployment settings
│
└── 🎯 optimized_models/            # Model files (place your models here)
    ├── *.pt                       # PyTorch models
    ├── *.onnx                     # ONNX models
    └── *.engine                   # TensorRT engines (Jetson only)
```

---

## 🚀 Quick Start

### 1. Setup Environment

**Create Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Verify Setup:**
```bash
python setup_verify.py
```

---

### 2. Configure Your Model

Edit `config/deployment_config.json`:

```json
{
  "model": {
    "path": "optimized_models/yolov8m_best.pt",  // Your model path
    "input_size": [800, 800],
    "num_classes": 4,
    "confidence_threshold": 0.35,
    "iou_threshold": 0.45
  },
  "inference": {
    "device": "cuda",  // Auto-detected (cuda/cpu)
    "half": true,      // FP16 precision
    "batch_size": 1
  }
}
```

**Supported Models:**
- YOLOv8s (Small) - Fastest, good accuracy
- YOLOv8m (Medium) - Better accuracy, moderate speed
- YOLOv8l (Large) - Best accuracy, slower
- Any custom YOLOv8 model trained on RDD2022

---

### 3. Run Detection

#### **Option 1: Web GUI (Recommended)**
```bash
streamlit run streamlit_app.py
# OR
start_gui.bat  # Windows
./start_gui.sh # Linux/Mac
```

**Features:**
- Upload images/videos for detection
- Real-time webcam detection
- Batch processing
- Interactive parameter tuning
- Download results (images/JSON)

#### **Option 2: Command Line**

**Image Detection:**
```bash
python detect_demo.py --model optimized_models/yolov8m_best.pt --source test.jpg
```

**Webcam Detection:**
```bash
python realtime_detection.py --model optimized_models/yolov8m_best.pt --source 0
```

**Batch Processing:**
```bash
python batch_test.py --model optimized_models/yolov8m_best.pt --source test_images/ --conf 0.35
```

#### **Option 3: Jetson Deployment**

See [Jetson Setup](#jetson-setup) section below.

---

## 🔧 Jetson Setup

### Prerequisites

**Hardware:**
- Jetson Nano (4GB) or Orin Nano (8GB)
- USB/CSI Camera
- GPS Module (optional)
- 64GB+ MicroSD card

**Software:**
- JetPack 5.1+ (or 4.6+ for Nano)
- Python 3.8+
- CUDA (included in JetPack)

### Installation

**1. Flash JetPack:**
```bash
# Use NVIDIA SDK Manager or pre-flashed SD card
# https://developer.nvidia.com/embedded/jetpack
```

**2. System Setup:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-dev build-essential \
    libopencv-dev python3-opencv cmake git

# Install GPS daemon (optional)
sudo apt install -y gpsd gpsd-clients python3-gps
```

**3. Install PyTorch (Jetson):**
```bash
# For JetPack 5.x (Orin Nano)
pip3 install torch torchvision

# For JetPack 4.x (Jetson Nano)
wget https://nvidia.box.com/shared/static/[wheel_link].whl
pip3 install torch-*.whl
```

**4. Install Project Dependencies:**
```bash
pip3 install ultralytics opencv-python numpy pandas \
    gpsd-py3 geopy sqlalchemy requests python-dotenv
```

**5. Copy Project to Jetson:**
```bash
# From your PC
scp -r Project/ jetson@<jetson-ip>:~/
```

### TensorRT Optimization

**Convert to TensorRT (on Jetson):**

```bash
cd ~/Project/jetson

# Option 1: FP16 (Recommended - Best balance)
./convert_tensorrt_fp16.sh
# Generates: optimized_models/yolov8m_fp16.engine

# Option 2: INT8 (Fastest - requires calibration)
./convert_tensorrt_int8.sh
# Generates: optimized_models/yolov8m_int8.engine
```

**Run Inference:**
```bash
# With TensorRT engine
python jetson/main_system.py --engine optimized_models/yolov8m_fp16.engine

# With PyTorch model (fallback)
python jetson/inference.py --model optimized_models/yolov8m_best.pt
```

### Performance Comparison

| Model | Jetson Nano | Orin Nano | Memory | Accuracy |
|-------|-------------|-----------|---------|----------|
| YOLOv8s PyTorch | 10-15 FPS | 25-30 FPS | 800MB | 100% baseline |
| YOLOv8s TensorRT FP16 | 20-25 FPS | 40-50 FPS | 500MB | 99% |
| YOLOv8m PyTorch | 5-8 FPS | 15-20 FPS | 1.2GB | 100% baseline |
| YOLOv8m TensorRT FP16 | 12-15 FPS | 25-35 FPS | 700MB | 99% |

---

## ⚙️ Configuration Guide

### deployment_config.json

```json
{
  "model": {
    "path": "optimized_models/yolov8m_best.pt",
    "input_size": [800, 800],          // Model input resolution
    "num_classes": 4,
    "class_names": [
      "Pothole", "Alligator Crack", 
      "Longitudinal Crack", "Other Damage"
    ],
    "confidence_threshold": 0.35,      // Min confidence (0.0-1.0)
    "iou_threshold": 0.45              // NMS threshold
  },
  
  "inference": {
    "device": "cuda",                   // cuda/cpu (auto-detected)
    "half": true,                       // FP16 precision
    "batch_size": 1                     // Batch size
  },
  
  "camera": {
    "source": 0,                        // Camera index or rtsp://
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  
  "gps": {
    "enabled": true,
    "port": "/dev/ttyUSB0",             // GPS serial port
    "baudrate": 9600
  },
  
  "reporting": {
    "enabled": true,
    "database": "detections.db",
    "min_confidence": 0.5,
    "report_interval": 300,             // Seconds
    
    "email_enabled": false,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "your_email@gmail.com",
    "recipient_email": "municipal@example.com",
    
    "webhook_enabled": false,
    "webhook_url": "https://api.example.com/road-anomalies"
  }
}
```

---

## 🔄 Using Multiple Models (YOLOv8s / YOLOv8m / Custom)

### Adding Your YOLOv8m Models

**1. Place model files in `optimized_models/`:**
```
optimized_models/
├── yolov8s_best.pt      # Small model
├── yolov8m_best.pt      # Medium model (YOUR NEW MODEL)
├── yolov8l_best.pt      # Large model (optional)
├── deployment_config.json
└── yolov8m_fp32.onnx    # ONNX export for TensorRT
```

**2. Update `config/deployment_config.json`:**
```json
{
  "model": {
    "path": "optimized_models/yolov8m_best.pt"  // Change to your model
  }
}
```

**3. Run detection:**
```bash
# CLI - specify model directly
python detect_demo.py --model optimized_models/yolov8m_best.pt --source test.jpg

# Or use config file
python jetson/inference.py  # Uses config/deployment_config.json
```

**4. Web GUI - automatic detection:**
- Streamlit app automatically detects all `.pt` and `.onnx` files in `optimized_models/`
- Select model from dropdown in sidebar

### Model Selection Guide

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|------------|-------|----------|----------|
| **YOLOv8s** | 11M | Fast | Good | Real-time, resource-limited |
| **YOLOv8m** | 25M | Medium | Better | Balanced performance |
| **YOLOv8l** | 43M | Slow | Best | Maximum accuracy |

**Recommendation:**
- **Jetson Nano**: Use YOLOv8s with TensorRT FP16
- **Jetson Orin Nano**: Use YOLOv8m with TensorRT FP16 (best balance)
- **Development PC**: Any model works

---

## 🎮 Device Detection

All scripts automatically detect GPU availability:

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**What you'll see:**
```bash
# PC with NVIDIA GPU:
🎮 Using GPU: NVIDIA GeForce RTX 3060

# PC without GPU:
⚠️  Using CPU (no GPU detected) - inference will be slower

# Jetson:
🎮 Using GPU: NVIDIA Tegra Orin
```

---

## 🛠️ Optimization Features

### Memory Optimization (Jetson-specific)

**Features Implemented:**
1. ✅ **Never load full dataset** - Streaming/generators only
2. ✅ **Resize early** - Images resized immediately after loading
3. ✅ **Normalize once** - Single normalization pass
4. ✅ **Use generators** - Memory-efficient iteration
5. ✅ **Measure memory** - Built-in memory monitoring

**Memory Monitor Usage:**
```python
from jetson.memory_monitor import MemoryMonitor

monitor = MemoryMonitor(enable_logging=True)
monitor.measure("Before inference")
# ... run inference ...
monitor.measure("After inference")
monitor.print_summary()
```

### Performance Tips

**For Jetson Nano (4GB):**
- Use YOLOv8s with TensorRT FP16
- Set `input_size: [640, 640]` for faster processing
- Enable `half: true` in config
- Use `batch_size: 1`

**For Jetson Orin Nano (8GB):**
- Use YOLOv8m with TensorRT FP16 for best accuracy
- Can handle `input_size: [800, 800]`
- Enable `half: true` in config
- Can use `batch_size: 2-4` for faster batch processing

---

## 📚 Application Guides

### Streamlit Web GUI

**Launch:**
```bash
streamlit run streamlit_app.py
```

**Features:**
- **Image Detection**: Upload images, view results, download annotations
- **Video Detection**: Process videos with progress tracking
- **Batch Analysis**: Upload multiple images at once
- **Webcam Detection**: Continuous real-time detection
- **Interactive Controls**: Adjust confidence, IoU thresholds live

**Parameters:**
- **Confidence Threshold** (0.35 default): Higher = fewer false positives
- **IoU Threshold** (0.45 default): Controls overlapping detection removal

### Command Line Tools

**detect_demo.py** - Simple detection:
```bash
python detect_demo.py --model optimized_models/yolov8m_best.pt \
                      --source test.jpg \
                      --conf 0.35 \
                      --save results/
```

**realtime_detection.py** - Real-time with GUI:
```bash
python realtime_detection.py --model optimized_models/yolov8m_best.pt \
                             --source 0 \
                             --conf 0.35
```

**batch_test.py** - Batch processing:
```bash
python batch_test.py --model optimized_models/yolov8m_best.pt \
                     --source test_images/ \
                     --conf 0.35 \
                     --save results/
```

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'ultralytics'**
```bash
pip install ultralytics
```

**2. CUDA not available on Jetson**
```bash
# Check CUDA installation
nvcc --version
# Reinstall PyTorch for Jetson (see Jetson Setup section)
```

**3. TensorRT conversion fails**
```bash
# Ensure you're on Jetson device
# TensorRT engines are hardware-specific
# First export to ONNX:
python -c "from ultralytics import YOLO; YOLO('optimized_models/yolov8m_best.pt').export(format='onnx')"
```

**4. GPS not working**
```bash
# Check GPS device
ls /dev/ttyUSB*
# Start GPS daemon
sudo systemctl start gpsd
```

**5. Out of memory on Jetson**
```bash
# Reduce input size in config
"input_size": [640, 640]  // Instead of [800, 800]
# Use smaller model (YOLOv8s instead of YOLOv8m)
```

**6. Model file not found**
```bash
# Check your model path in config
ls optimized_models/
# Update path in config/deployment_config.json
```

---

## 📈 Expected Performance

### Accuracy Metrics
- **mAP50**: 0.75-0.85 (depending on model size)
- **Precision**: ~0.80
- **Recall**: ~0.75

### Speed Benchmarks
- **PC (CPU)**: 2-5 FPS
- **PC (GPU)**: 30-60 FPS
- **Jetson Nano**: 10-25 FPS (TensorRT)
- **Jetson Orin Nano**: 25-50 FPS (TensorRT)

---
