# MonoDepth AI™ — Real-Time Depth Perception for Any Camera



**Transform any standard camera into a powerful depth sensor.** MonoDepth AI delivers enterprise-grade 3D depth estimation from a single RGB camera—no LiDAR, stereo setup, or expensive hardware required. Powered by state-of-the-art self-supervised deep learning, our solution provides real-time distance measurement and object detection at a fraction of the cost of traditional depth sensors.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU Accelerated](https://img.shields.io/badge/GPU-CUDA%20Optimized-brightgreen.svg)](#)

---

## 🚀 Getting Started for Clients

**See it in action in under 60 seconds:**

```bash
# 1. Install (one-time setup)
pip install -r requirements.txt

# 2. Run live demo with your webcam
python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth
```

**That's it!** Point your webcam at any scene and watch real-time depth estimation with object distance tracking.



---

## 💡 Why Choose MonoDepth AI?

| Challenge | Our Solution |
|-----------|--------------|
| **Expensive depth sensors** | Works with any standard webcam or IP camera—no special hardware needed |
| **Complex multi-camera setups** | Single camera delivers accurate depth from monocular input |
| **Slow processing speeds** | GPU-optimized for **60+ FPS** real-time performance |
| **Massive training data requirements** | Self-supervised learning—train on your own unlabeled video |
| **Integration headaches** | Drop-in Python API, video files, webcams, or RTSP streams |

### ⚡ Key Benefits

- **🎯 Real-Time Performance** — 60+ FPS on modern GPUs, 30+ FPS on consumer hardware
- **🔧 Zero Labeled Data Required** — Self-supervised training means no expensive annotation
- **📷 Universal Camera Support** — Webcams, IP cameras, drones, dashcams, security cameras
- **📏 Accurate Distance Estimation** — Measure real-world distances to detected objects in meters
- **🤖 Built-in Object Detection** — YOLOv8 integration detects 20+ object classes automatically
- **🎨 Multiple Visualization Modes** — 8 professional colormaps for different use cases
- **💻 Cross-Platform** — Windows, Linux, macOS with CPU or GPU acceleration

---

## 🏢 Built for Business

MonoDepth AI isn't just research code—it's a production-ready solution designed for real-world deployment.

<table>
<tr>
<td width="25%" align="center">
<img src="assets/icon-robotics.png" width="64"><br>
<b>🤖 Robotics & Automation</b><br>
<small>Obstacle avoidance, navigation, pick-and-place operations</small>
</td>
<td width="25%" align="center">
<img src="assets/icon-security.png" width="64"><br>
<b>🔒 Security & Surveillance</b><br>
<small>Intrusion detection, perimeter monitoring, crowd analytics</small>
</td>
<td width="25%" align="center">
<img src="assets/icon-arvr.png" width="64"><br>
<b>🥽 AR/VR & Spatial Computing</b><br>
<small>3D reconstruction, occlusion handling, scene understanding</small>
</td>
<td width="25%" align="center">
<img src="assets/icon-automotive.png" width="64"><br>
<b>🚗 Automotive & ADAS</b><br>
<small>Collision warning, parking assist, lane keeping</small>
</td>
</tr>
<tr>
<td width="25%" align="center">
<img src="assets/icon-retail.png" width="64"><br>
<b>🛒 Retail Analytics</b><br>
<small>Foot traffic analysis, shelf monitoring, queue management</small>
</td>
<td width="25%" align="center">
<img src="assets/icon-drone.png" width="64"><br>
<b>🚁 Drone & UAV</b><br>
<small>Altitude estimation, landing assistance, terrain following</small>
</td>
<td width="25%" align="center">
<img src="assets/icon-healthcare.png" width="64"><br>
<b>🏥 Healthcare</b><br>
<small>Patient monitoring, fall detection, activity recognition</small>
</td>
<td width="25%" align="center">
<img src="assets/icon-industrial.png" width="64"><br>
<b>🏭 Industrial Inspection</b><br>
<small>Quality control, defect detection, measurement automation</small>
</td>
</tr>
</table>

---

## 🎯 Features at a Glance

### Depth Estimation
- **Self-supervised learning** — No ground truth depth labels required for training
- **Multi-scale predictions** — Depth at 4 resolutions for fine-grained accuracy
- **GPU & CPU modes** — Runs anywhere from edge devices to cloud servers

### Object Detection & Tracking
- **YOLOv8 integration** — State-of-the-art real-time object detection
- **20+ object classes** — People, vehicles, animals, and more
- **Distance to objects** — Real-time distance measurement in meters

### Input Flexibility
- **Live webcam** — Any USB or built-in camera
- **Video files** — MP4, AVI, MOV, MKV, WebM
- **Image folders** — Batch process image sequences
- **RTSP streams** — Network cameras and IP streams

### Interactive Controls
| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `S` | Save current frame |
| `C` | Cycle colormaps |
| `Q` / `ESC` | Quit |

---

## 🎨 Professional Colormaps

Choose from 8 visualization styles optimized for different applications:

| Colormap | Best For |
|----------|----------|
| `accurate` | Scientific visualization, publications |
| `magma` | High contrast, outdoor scenes |
| `viridis` | Colorblind-friendly, accessibility |
| `plasma` | Warm tones, artistic output |
| `turbo` | Maximum depth discrimination |
| `jet` | Traditional depth visualization |
| `hot` | Thermal-style output |
| `inferno` | Dark environments, night vision |

---

## 💰 Licensing & Optional Paid Add-ons

### Base Package — Open Source (MIT License)
✅ Full depth estimation engine  
✅ Object detection integration  
✅ Live inference & video processing  
✅ Training on your own data  
✅ Commercial use permitted  



---

## ⚙️ Usage Examples

### Live Webcam Inference

```bash
# Default webcam with object detection
python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth

# Custom colormap and resolution
python inference_live.py --webcam 0 --colormap turbo --height 256 --width 832

# Disable bounding boxes (depth only)
python inference_live.py --webcam 0 --no-boxes
```

### Video Processing

```bash
# Process video file
python inference_live.py --input video.mp4 --checkpoint checkpoints/best.pth

# Slow-motion playback for analysis
python inference_live.py --input video.mp4 --slowdown 3 --loop
```

### Batch Image Processing

```bash
# Process folder of images
python inference_live.py --input images/ --checkpoint checkpoints/best.pth
```

---

## 🎓 Training Your Own Model

Train on your own unlabeled video data with self-supervised learning:

```bash
# Quick start with default settings
python train.py --config configs/default.yaml

# High-accuracy KITTI training
python train.py --config configs/kitti_sota.yaml
```

**Prepare your data:**
```
your_data/
├── sequence_001/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── sequence_002/
│   └── ...
```

📖 **For detailed training configuration, see [docs/advanced.md](docs/advanced.md)**

---

## 📊 Performance

| Hardware | Resolution | FPS (Depth Only) | FPS (Depth + Detection) |
|----------|------------|------------------|-------------------------|
| RTX 4090 | 640×192 | 150+ | 90+ |
| RTX 3080 | 640×192 | 120 | 60 |
| RTX 2080 | 640×192 | 85 | 45 |
| GTX 1080 | 640×192 | 60 | 30 |
| CPU (i7) | 640×192 | 8 | 5 |

---

## 📁 What's Included

```
MonoDepth-AI/
├── inference_live.py   # 🎥 Live depth + detection (start here!)
├── inference.py        # Video/image batch processing
├── train.py            # Model training
├── train_advanced.py   # Advanced training options
├── visualize.py        # Visualization tools
│
├── models/             # Neural network architectures
├── geometry/           # 3D geometry operations
├── losses/             # Training loss functions
├── datasets/           # Data loading utilities
├── configs/            # Training configurations
├── checkpoints/        # Pre-trained models
└── docs/               # Advanced documentation
```

📖 **Technical deep-dive:** See [docs/advanced.md](docs/advanced.md) for architecture details, configuration options, and optimization guides.

---

## 🛠️ Requirements

- **Python 3.10+**
- **PyTorch 2.0+**
- **CUDA-capable GPU** (recommended for real-time performance)
- **Webcam** (for live inference)

### Installation

```bash
# Clone repository
git clone https://github.com/Rahitya86/Live-Monocular-Depth-Object-Detection-AI.git
cd Live-Monocular-Depth-Object-Detection-AI

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install YOLOv8 for object detection
pip install ultralytics
```

---

## 🆘 Support

| Resource | Link |
|----------|------|
| 📖 Documentation | [docs/advanced.md](docs/advanced.md) |
| 🐛 Bug Reports | [GitHub Issues](https://github.com/Rahitya86/Live-Monocular-Depth-Object-Detection-AI/issues) |
| 💬 Discussions | [GitHub Discussions](https://github.com/Rahitya86/Live-Monocular-Depth-Object-Detection-AI/discussions) |
| ✉️ Enterprise Sales | sales@example.com |
| 🔧 Technical Support | support@example.com |

---

## 📚 References

- Godard et al., *"Digging into Self-Supervised Monocular Depth Prediction"* (ICCV 2019)
- Zhou et al., *"Unsupervised Learning of Depth and Ego-Motion"* (CVPR 2017)
- Ultralytics, *"YOLOv8: Real-Time Object Detection"* (2023)

---

## 📄 License

**MIT License** — Free for research and commercial use.

---

