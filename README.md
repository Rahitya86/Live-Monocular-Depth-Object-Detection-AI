# Live Monocular Depth Estimation with Object Detection

Real-time depth estimation and object detection from a single camera. Combines self-supervised monocular depth learning with YOLOv8/Faster R-CNN for live scene understanding.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

### Depth Estimation
- **Self-supervised learning**: No depth ground truth required during training
- **Real-time inference**: 15-30 FPS on GPU, 5-10 FPS on CPU
- **Multi-scale predictions**: Depth at multiple resolutions (1/1, 1/2, 1/4, 1/8)
- **Multiple colormaps**: Accurate, Magma, Viridis, Plasma, Jet, Turbo, Hot

### Object Detection
- **YOLOv8 Integration**: Fast and accurate object detection (default)
- **Faster R-CNN Fallback**: Works without ultralytics installed
- **20+ Object Classes**: Person, car, truck, bus, motorcycle, bicycle, dog, cat, etc.
- **Distance Estimation**: Real-time distance to detected objects in meters

### Live Inference
- **Webcam support**: Real-time processing from any webcam
- **Video files**: Process MP4, AVI, MOV, MKV, WebM
- **Image sequences**: Batch process folders of images
- **Interactive controls**: Pause, save frames, cycle colormaps

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LIVE INFERENCE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Camera/Video ──> Frame ──┬──> DepthNet ──> Depth Map ──┐              │
│                            │                              │              │
│                            └──> YOLOv8 ──> Detections ───┤              │
│                                                           ▼              │
│                                              ┌─────────────────────┐    │
│                                              │  Combined Output:   │    │
│                                              │  - Depth colormap   │    │
│                                              │  - Bounding boxes   │    │
│                                              │  - Distance labels  │    │
│                                              └─────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      SELF-SUPERVISED TRAINING                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Frame t-1 ─┐                                                           │
│             │                                                           │
│  Frame t ───┼──> DepthNet ──> Depth Map ─┐                              │
│             │                             │                              │
│  Frame t+1 ─┘                             ▼                              │
│             │                    Inverse Warping                        │
│             │                             │                              │
│             └──> PoseNet ──> Poses ───────┤                              │
│                                           ▼                              │
│                               Photometric Loss + Auto-mask              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Live-Monocular-Depth-Object-Detection-AI/
├── inference_live.py   # 🎥 Live depth + object detection (main script)
├── inference.py        # Video/image inference
├── train.py            # Training script
├── train_advanced.py   # Advanced training with more options
├── visualize.py        # Visualization utilities
├── smoke_test.py       # Verify installation
│
├── models/
│   ├── encoder.py      # ResNet encoder backbone
│   ├── decoder.py      # Multi-scale depth decoder
│   ├── depth_net.py    # Complete DepthNet model
│   └── pose_net.py     # 6-DoF camera pose estimation
│
├── geometry/
│   ├── camera.py       # Camera intrinsics & projection
│   ├── transform.py    # Pose transformations
│   └── warping.py      # Differentiable inverse warping
│
├── losses/
│   ├── ssim.py         # Structural Similarity loss
│   ├── photometric.py  # Photometric reconstruction loss
│   ├── smoothness.py   # Edge-aware depth smoothness
│   ├── combined.py     # Combined multi-scale loss
│   ├── stereo.py       # Stereo consistency loss
│   └── temporal.py     # Temporal consistency loss
│
├── datasets/
│   ├── mono_dataset.py # Monocular video dataset
│   ├── kitti_dataset.py# KITTI dataset loader
│   └── augmentation.py # Data augmentation
│
├── configs/            # Training configurations
├── test_images/        # Sample test images
└── requirements.txt    # Dependencies
```

## 🚀 Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for real-time performance)
- Webcam (for live inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/Rahitya86/Live-Monocular-Depth-Object-Detection-AI.git
cd Live-Monocular-Depth-Object-Detection-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install YOLOv8 (optional but recommended)
pip install ultralytics
```

## 🎮 Quick Start

### 1. Live Webcam Inference (with Object Detection)

```bash
# Run with webcam (default settings)
python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth

# With custom colormap
python inference_live.py --webcam 0 --colormap magma

# Disable object detection boxes
python inference_live.py --webcam 0 --no-boxes
```

### 2. Video File Processing

```bash
# Process a video file
python inference_live.py --input video.mp4 --checkpoint checkpoints/best.pth

# Slow motion playback
python inference_live.py --input video.mp4 --slowdown 2

# Loop playback
python inference_live.py --input video.mp4 --loop
```

### 3. Image Folder Processing

```bash
# Process folder of images
python inference_live.py --input images_folder/ --checkpoint checkpoints/best.pth
```

### Interactive Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `S` | Save current frame |
| `C` | Cycle through colormaps |
| `Q` / `ESC` | Quit |

## 🎨 Colormaps

| Colormap | Description |
|----------|-------------|
| `accurate` | Matplotlib magma (most accurate, default) |
| `magma` | OpenCV magma |
| `viridis` | Perceptually uniform |
| `plasma` | Warm colors |
| `inferno` | Dark to bright |
| `jet` | Classic rainbow |
| `turbo` | Improved rainbow |
| `hot` | Black to white via red |

## 🔧 Command Line Options

```bash
python inference_live.py [OPTIONS]

Input Options:
  --input PATH          Input video file, image, or folder
  --webcam ID           Webcam device ID (0, 1, 2, ...)

Model Options:
  --checkpoint PATH     Path to model checkpoint (default: checkpoints/best.pth)
  --encoder NAME        Encoder architecture (default: resnet18)
  --yolo MODEL          YOLO model file (default: yolov8n.pt)

Display Options:
  --colormap NAME       Depth colormap (default: accurate)
  --height H            Input height (default: 192)
  --width W             Input width (default: 640)
  --no-boxes            Disable bounding boxes
  --no-distance         Disable distance labels

Playback Options:
  --slowdown N          Playback slowdown factor
  --loop                Loop video/image playback
  --detector-thresh F   Detection confidence threshold (default: 0.35)
  --scaling-factor F    Depth scaling factor (default: 10.0)
```

## 🎓 Training Your Own Model

### Loss Functions

1. **Photometric Loss**: L1 + SSIM between target and warped source images
2. **Auto-masking**: Excludes static pixels automatically
3. **Edge-aware Smoothness**: Depth regularization respecting image edges
4. **Multi-scale**: Loss computed at 4 scales for better convergence

### Train with KITTI Dataset

```bash
# Download KITTI raw data and run:
python train.py --config configs/kitti_sota.yaml

# Or use lightweight config for faster training:
python train.py --config configs/kitti_lightweight.yaml
```

### Train with Custom Data

Prepare your data:
```
data/
├── sequence_001/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── sequence_002/
│   └── ...
```

```bash
python train.py --config configs/default.yaml
```

### Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  encoder: "resnet18"
  min_depth: 0.1
  max_depth: 100.0

training:
  learning_rate: 1.0e-4
  ssim_weight: 0.85
  smoothness_weight: 0.001
  use_auto_mask: true
```

## 🔍 Object Detection Details

### Supported Classes

The system detects and tracks these objects with distance estimation:

| Category | Objects |
|----------|---------|
| **People** | Person |
| **Vehicles** | Car, Truck, Bus, Motorcycle, Bicycle, Train, Boat, Airplane |
| **Animals** | Dog, Cat, Horse, Cow, Sheep, Bird, Elephant, Bear, Zebra, Giraffe |

### Detection Backends

1. **YOLOv8** (recommended): Fast, accurate, requires `ultralytics`
2. **Faster R-CNN**: Automatic fallback if YOLO unavailable

### Distance Estimation

- Uses median depth within bounding box
- Temporal smoothing for stable readings
- Configurable scaling factor for calibration

## 🧠 Model Details

### DepthNet

- **Encoder**: ResNet-18 backbone (11M params)
- **Decoder**: Progressive upsampling with skip connections
- **Output**: Multi-scale disparity maps (4 scales)
- **Depth Range**: 0.1m to 100m

### PoseNet

- **Architecture**: Lightweight CNN encoder
- **Output**: 6-DoF camera pose (rotation + translation)
- **Use**: Self-supervised training only

### Object Detector

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| YOLOv8n | ~45 FPS | Good | 6 MB |
| YOLOv8s | ~35 FPS | Better | 22 MB |
| Faster R-CNN | ~15 FPS | Best | 160 MB |

## ⚡ Performance

| Hardware | FPS (Depth Only) | FPS (Depth + Detection) |
|----------|------------------|-------------------------|
| RTX 3080 | ~60 FPS | ~30 FPS |
| RTX 2060 | ~45 FPS | ~25 FPS |
| GTX 1060 | ~30 FPS | ~15 FPS |
| CPU (i7) | ~8 FPS | ~5 FPS |

## 📚 References

- **Monodepth2**: Digging into Self-Supervised Monocular Depth Prediction
- **SfMLearner**: Unsupervised Learning of Depth and Ego-Motion
- **YOLOv8**: Ultralytics Real-Time Object Detection

## 📄 License

MIT License - Free for research and commercial use.

## 🤝 Contributing

Contributions welcome! Ideas:
- Additional encoder backbones (EfficientNet, ConvNeXt)
- More object detection models
- ONNX/TensorRT export
- Mobile deployment
- Stereo depth estimation
