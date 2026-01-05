# Monocular Depth Estimation (Tesla-Style Self-Supervised)

A complete, runnable implementation of self-supervised monocular depth estimation, inspired by Tesla's approach to depth perception for autonomous driving.

## Overview

This project implements a self-supervised depth estimation system that learns to predict depth from single images without requiring ground-truth depth labels. The key idea is to use photometric consistency between consecutive video frames as the training signal.

### Key Features

- **Self-supervised learning**: No depth ground truth required
- **Multi-scale predictions**: Depth at multiple resolutions (1/1, 1/2, 1/4, 1/8)
- **Auto-masking**: Handles static objects and occlusions
- **Edge-aware smoothness**: Depth regularization respecting image edges
- **Differentiable warping**: End-to-end trainable geometry operations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Supervised Training                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frame t-1 ─┐                                                │
│             │                                                │
│  Frame t ───┼──> DepthNet ──> Depth Map ─┐                  │
│             │                             │                  │
│  Frame t+1 ─┘                             ▼                  │
│             │                    Inverse Warping            │
│             │                             │                  │
│             └──> PoseNet ──> Poses ───────┤                  │
│                                           ▼                  │
│                               Photometric Loss + Auto-mask  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
monodepth-starter/
├── models/
│   ├── encoder.py      # ResNet encoder (TODO: swap for EfficientNet, etc.)
│   ├── decoder.py      # Multi-scale depth decoder
│   ├── depth_net.py    # Complete DepthNet
│   └── pose_net.py     # 6-DoF PoseNet
├── geometry/
│   ├── camera.py       # Intrinsics, projection, backprojection
│   ├── transform.py    # Pose transformations
│   └── warping.py      # Differentiable inverse warping
├── losses/
│   ├── ssim.py         # Structural Similarity loss
│   ├── photometric.py  # Photometric reconstruction loss
│   ├── smoothness.py   # Edge-aware smoothness
│   └── combined.py     # Combined multi-scale loss
├── datasets/
│   └── mono_dataset.py # Dataset for video frame triplets
├── utils/
│   └── helpers.py      # Training utilities
├── configs/
│   └── default.yaml    # Training configuration
├── train.py            # Training script
├── visualize.py        # Visualization script
├── smoke_test.py       # Smoke test (validates everything works)
└── requirements.txt    # Python dependencies
```

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone or create the project
cd monodepth-starter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Smoke Test

Verify everything works with synthetic data:

```bash
python smoke_test.py
```

This will:
- Test geometry operations
- Test model forward passes
- Test loss computation
- Run 1 training step
- Run 1 inference step

### 2. Train with Synthetic Data

Quick training test:

```bash
python train.py --config configs/default.yaml --synthetic
```

### 3. Train with Real Data

Prepare your data in the following structure:
```
data/
├── sequence_001/
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── sequence_002/
│   └── ...
```

Then train:

```bash
python train.py --config configs/default.yaml
```

### 4. Visualize Results

```bash
python visualize.py --image path/to/image.jpg --checkpoint checkpoints/best.pth
```

## Training Details

### Loss Functions

1. **Photometric Loss**: L1 + SSIM between target and warped source images
2. **Auto-masking**: Excludes pixels where warped image is not better than original
3. **Edge-aware Smoothness**: Regularizes depth to be smooth except at image edges
4. **Multi-scale**: Loss computed at 4 scales for better convergence

### Hyperparameters

Key parameters in `configs/default.yaml`:

```yaml
model:
  encoder: "resnet18"  # Backbone network
  min_depth: 0.1       # Minimum depth (meters)
  max_depth: 100.0     # Maximum depth (meters)

training:
  learning_rate: 1.0e-4
  ssim_weight: 0.85    # Weight for SSIM vs L1
  smoothness_weight: 0.001
  use_auto_mask: true  # Enable auto-masking
```

## Customization

### TODO: Swap Encoder

In `models/encoder.py`, you can replace the ResNet encoder:

```python
# Currently supported:
encoder = get_encoder('resnet18', pretrained=True)
encoder = get_encoder('resnet34', pretrained=True)
encoder = get_encoder('resnet50', pretrained=True)

# TODO: Add support for:
# - EfficientNet
# - ConvNeXt
# - Vision Transformer (ViT)
```

### TODO: Swap Dataset

In `datasets/mono_dataset.py`, modify for your data format:

```python
# Currently expects:
# - Folder structure with sequences
# - PNG images

# TODO: Add support for:
# - KITTI format
# - Cityscapes format
# - Custom video formats
```

## Model Details

### DepthNet

- **Encoder**: ResNet-18 (11M params) producing 5 feature scales
- **Decoder**: Progressive upsampling with skip connections
- **Output**: Multi-scale disparity maps converted to depth

### PoseNet

- **Architecture**: Simple CNN encoder
- **Output**: 6-DoF pose (3 axis-angle rotation + 3 translation)
- **Scale**: Predictions are scaled by 0.01 for stability

### Geometry

- **Inverse Warping**: Projects target pixels to 3D using depth, transforms to source camera, projects back to 2D
- **Grid Sampling**: Bilinear interpolation for differentiable warping

## Performance Tips

1. **GPU Memory**: Reduce batch size or image resolution if OOM
2. **Training Speed**: Use `num_workers > 0` for data loading
3. **Stability**: The smoothness loss helps prevent depth collapse
4. **Multi-GPU**: Wrap models in `DataParallel` for multi-GPU training

## References

This implementation is inspired by:

1. **Monodepth2** - Digging into Self-Supervised Monocular Depth Prediction
2. **SfMLearner** - Unsupervised Learning of Depth and Ego-Motion
3. **PackNet-SfM** - 3D Packing for Self-Supervised Monocular Depth Estimation

## License

MIT License - Feel free to use for research and commercial applications.

## Contributing

Contributions welcome! Areas for improvement:
- Additional encoder backbones
- More dataset formats
- Evaluation metrics (AbsRel, SqRel, RMSE, etc.)
- Pre-trained weights
- ONNX/TensorRT export
