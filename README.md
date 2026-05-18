# MonoDepth AI™ — Real-Time Depth Perception for Any Camera

<p align="center">
  <img src="assets/hero-banner.png" alt="MonoDepth AI Demo" width="100%">
</p>

**Transform any standard camera into a powerful depth sensor.** MonoDepth AI delivers 3D depth estimation from a single RGB camera—no LiDAR, stereo setup, or expensive hardware required. The project combines self-supervised monocular depth learning with real-time object detection for practical deployment and research.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Getting Started (Quick)

Run a live demo with a webcam using the provided pre-trained checkpoint:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth
```

To train (example):

```bash
python train.py --config configs/default.yaml
```

## Maintainer

- Name: Rahitya
- Email: ulapallirahitya@gmail.com
- GitHub: @Rahitya86

## Overview

This repository implements a self-supervised monocular depth estimation pipeline with integrated object detection and distance estimation. It is intended for reproducible experiments and practical live inference.

Key capabilities:
- Self-supervised training (photometric + SSIM + smoothness)
- Multi-scale depth outputs and pose estimation
- Real-time inference pipeline with YOLOv8 object detection integration
- Support for webcams, video files, image folders, and RTSP streams

## Features

- Self-supervised monocular depth learning (no ground-truth depth required)
- Multi-scale disparity → depth conversion with configurable min/max depth
- PoseNet for relative pose estimation during training
- YOLOv8 detection (recommended) with torchvision Faster R-CNN fallback
- Distance estimation per detected object (median depth inside bounding box)
- Interactive live demo controls (pause, save frame, cycle colormap)

## Project Structure

```
./
├── inference_live.py   # Live depth + object detection demo
├── inference.py        # Batch inference for video/images
├── train.py            # Training entrypoint
├── train_advanced.py   # Advanced training options
├── visualize.py        # Visualization utilities
├── models/             # Encoder/Decoder/PoseNet implementations
├── geometry/           # Camera, warping, projection utilities
├── losses/             # Photometric, smoothness, combined losses
├── datasets/           # KITTI and generic video dataset loaders
├── configs/            # YAML configs for reproducible runs
├── checkpoints/        # Pre-trained model checkpoints
└── docs/               # Advanced documentation and training details
```

## Usage Examples

Live webcam with object detection:

```bash
python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth
```

Depth-only live (no boxes):

```bash
python inference_live.py --webcam 0 --no-boxes --checkpoint checkpoints/best.pth
```

Process a video file:

```bash
python inference_live.py --input video.mp4 --checkpoint checkpoints/best.pth
```

Batch process an image folder:

```bash
python inference_live.py --input images_folder/ --checkpoint checkpoints/best.pth
```

## Training

Train with the default config:

```bash
python train.py --config configs/default.yaml
```

High-accuracy KITTI config:

```bash
python train.py --config configs/kitti_sota.yaml
```

See `docs/advanced.md` for detailed hyperparameters and loss formulations.

## Evaluation

Depth metrics implemented: AbsRel, SqRel, RMSE, RMSE(log), δ thresholds. Use provided evaluation scripts when ground truth is available (KITTI-format).

## Configuration

Configurations are YAML files in `configs/`. They define model, training, and data parameters. Use these configs to reproduce experiments and tune models.

## Assets & Visuals

This repository supports optional visual assets. To include screenshots or GIFs in the README, place them in the `assets/` directory and reference them in the Markdown.

## Support & Contact

- Maintainer: ulapallirahitya@gmail.com
- Issues & discussions: https://github.com/Rahitya86/Live-Monocular-Depth-Object-Detection-AI/issues

## License

MIT License — see `LICENSE` for details.

---

