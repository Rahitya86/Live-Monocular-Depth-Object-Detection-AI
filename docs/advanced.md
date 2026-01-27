# Advanced Configuration & Technical Reference

This document contains detailed technical specifications and advanced configuration options for developers and power users.

## 🔧 Complete Configuration Reference

### Model Configuration

```yaml
model:
  encoder: "resnet18"           # Options: resnet18, resnet34, resnet50
  pretrained_encoder: true      # Use ImageNet pretrained weights
  min_depth: 0.1                # Minimum depth in meters
  max_depth: 100.0              # Maximum depth in meters
  num_layers: 18                # Encoder depth (18, 34, 50)
  pose_model_type: "separate"   # shared or separate pose encoder
```

### Training Configuration

```yaml
training:
  batch_size: 12
  learning_rate: 1.0e-4
  scheduler_step_size: 15
  num_epochs: 20
  height: 192
  width: 640
  frame_ids: [0, -1, 1]         # Frame indices for temporal context
  num_scales: 4                 # Multi-scale predictions (1, 1/2, 1/4, 1/8)
  
  # Loss weights
  ssim_weight: 0.85             # Structural similarity weight
  smoothness_weight: 0.001      # Edge-aware smoothness weight
  
  # Auto-masking
  use_auto_mask: true           # Enable automatic masking of static pixels
  avg_reprojection: false       # Use average vs minimum reprojection
  
  # Augmentation
  augment: true
  flip_prob: 0.5
  color_aug_prob: 0.5
```

### Data Configuration

```yaml
data:
  dataset: "kitti"              # Dataset type
  data_path: "./datasets/"      # Root path to dataset
  split: "eigen_zhou"           # Train/val split
  png: false                    # Use PNG instead of JPG
  num_workers: 8                # DataLoader workers
```

## 📁 Project Architecture

### Module Details

```
models/
├── encoder.py      # ResNet encoder backbone
│   └── ResnetEncoder: Extracts multi-scale features [1/1, 1/2, 1/4, 1/8, 1/16]
│
├── decoder.py      # Multi-scale depth decoder
│   └── DepthDecoder: Progressive upsampling with skip connections
│
├── depth_net.py    # Complete DepthNet model
│   └── DepthNet: Combines encoder + decoder + disparity-to-depth
│
└── pose_net.py     # 6-DoF camera pose estimation
    └── PoseNet: Outputs axis-angle rotation + translation

geometry/
├── camera.py       # Camera intrinsics & projection
│   ├── CameraIntrinsics: Focal length, principal point
│   ├── BackprojectDepth: 2D pixels → 3D points
│   └── Project3D: 3D points → 2D pixels
│
├── transform.py    # Pose transformations
│   ├── axis_angle_to_matrix: Rodrigues formula
│   └── transformation_from_parameters: Build 4x4 matrix
│
└── warping.py      # Differentiable inverse warping
    └── InverseWarping: Reconstruct source from target + depth + pose

losses/
├── ssim.py         # Structural Similarity Index
│   └── SSIM: Window-based structural similarity
│
├── photometric.py  # Photometric reconstruction loss
│   └── PhotometricLoss: L1 + SSIM combined
│
├── smoothness.py   # Edge-aware depth smoothness
│   └── SmoothnessLoss: Gradient-weighted depth regularization
│
├── combined.py     # Combined multi-scale loss
│   └── CombinedLoss: Aggregates all loss components
│
├── stereo.py       # Stereo consistency loss (optional)
│   └── StereoConsistencyLoss: Left-right depth consistency
│
└── temporal.py     # Temporal consistency loss (optional)
    └── TemporalConsistencyLoss: Frame-to-frame depth consistency
```

## 🎓 Training Deep Dive

### Loss Functions Explained

#### 1. Photometric Loss
Measures reconstruction quality between target and warped source images:
```
L_photo = α * SSIM(I_t, I_s→t) + (1-α) * |I_t - I_s→t|
```
Where α = 0.85 (ssim_weight)

#### 2. Auto-masking
Automatically excludes static pixels:
```
M_auto = [L(I_t, I_s→t) < L(I_t, I_s)]
```
Ignores regions where warping doesn't improve reconstruction.

#### 3. Edge-aware Smoothness
Depth regularization respecting image edges:
```
L_smooth = |∂_x d*| * e^(-|∂_x I|) + |∂_y d*| * e^(-|∂_y I|)
```
Where d* = d / mean(d) (normalized depth)

#### 4. Multi-scale Loss
Total loss computed at 4 scales with equal weighting:
```
L_total = Σ_s (L_photo_s + λ * L_smooth_s) / 4
```

### Training Tips

1. **Learning Rate**: Start with 1e-4, decay by 0.1 after 15 epochs
2. **Batch Size**: 12-16 works well for ResNet18 on 11GB GPU
3. **Resolution**: 640×192 is good balance of speed/accuracy
4. **Epochs**: 20 epochs sufficient for KITTI, more for custom data
5. **Augmentation**: Always enable for better generalization

### Custom Dataset Preparation

```python
# Expected folder structure:
data/
├── train/
│   ├── sequence_001/
│   │   ├── frame_0000.png
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── sequence_002/
│   │   └── ...
├── val/
│   └── ...

# Requirements:
# - Consecutive frames from video sequences
# - At least 3 frames per sequence (for temporal context)
# - Consistent resolution within sequence
# - Camera intrinsics file (optional, for accurate metric depth)
```

### Camera Intrinsics

For accurate metric depth, provide camera intrinsics:
```yaml
camera:
  fx: 718.856    # Focal length x
  fy: 718.856    # Focal length y
  cx: 607.193    # Principal point x
  cy: 185.216    # Principal point y
```

## 🔍 Object Detection Configuration

### YOLOv8 Model Variants

| Model | Size | mAP | Speed (GPU) |
|-------|------|-----|-------------|
| yolov8n.pt | 6.3 MB | 37.3 | 0.9 ms |
| yolov8s.pt | 22.5 MB | 44.9 | 1.2 ms |
| yolov8m.pt | 52.0 MB | 50.2 | 2.6 ms |
| yolov8l.pt | 87.7 MB | 52.9 | 4.3 ms |

### Detection Threshold Tuning

```bash
# Higher threshold = fewer false positives
python inference_live.py --detector-thresh 0.5

# Lower threshold = more detections
python inference_live.py --detector-thresh 0.25
```

### Depth Scaling Calibration

The `--scaling-factor` converts disparity to real-world distance:
```bash
# Default scaling (works for most webcams)
python inference_live.py --scaling-factor 10.0

# Calibrate with known distance object:
# 1. Place object at known distance (e.g., 2m)
# 2. Adjust scaling until displayed distance matches
```

## ⚡ Performance Optimization

### GPU Memory Usage

| Resolution | Batch 1 | Batch 4 | Batch 8 |
|------------|---------|---------|---------|
| 640×192 | ~1.5 GB | ~3 GB | ~5 GB |
| 1024×320 | ~2.5 GB | ~6 GB | ~10 GB |
| 1280×384 | ~3.5 GB | ~8 GB | N/A |

### Speed Benchmarks

| Configuration | RTX 3080 | RTX 2080 | GTX 1080 |
|---------------|----------|----------|----------|
| DepthNet only | 120 FPS | 85 FPS | 60 FPS |
| Depth + YOLO | 60 FPS | 45 FPS | 30 FPS |
| CPU mode | 8 FPS | 6 FPS | 5 FPS |

### Optimization Flags

```bash
# Use FP16 for faster inference (slight accuracy loss)
python inference_live.py --fp16

# Reduce resolution for speed
python inference_live.py --height 128 --width 416

# Use lightweight YOLO model
python inference_live.py --yolo yolov8n.pt
```

## 🧪 Evaluation Metrics

### Depth Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Abs Rel | mean(\|d - d*\| / d*) | Absolute relative error |
| Sq Rel | mean((d - d*)² / d*) | Squared relative error |
| RMSE | sqrt(mean((d - d*)²)) | Root mean square error |
| RMSE log | sqrt(mean((log d - log d*)²)) | Log-scale RMSE |
| δ < 1.25 | % of max(d/d*, d*/d) < 1.25 | Threshold accuracy |

### Expected KITTI Results

| Model | Abs Rel | Sq Rel | RMSE | δ < 1.25 |
|-------|---------|--------|------|----------|
| ResNet18 | 0.115 | 0.903 | 4.863 | 0.877 |
| ResNet50 | 0.106 | 0.806 | 4.630 | 0.893 |

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or resolution
python train.py --batch_size 4 --height 128 --width 416
```

**Slow Training**
```bash
# Increase workers, use mixed precision
python train.py --num_workers 8 --amp
```

**NaN Losses**
- Reduce learning rate to 5e-5
- Enable gradient clipping
- Check for corrupt images in dataset

**Poor Depth Quality**
- Ensure sufficient motion between frames
- Avoid pure rotation sequences
- Check camera intrinsics accuracy

### Debug Mode

```bash
# Verbose output with intermediate visualizations
python train.py --debug --save_frequency 100
```

## 📚 References

- Godard et al., "Digging into Self-Supervised Monocular Depth Prediction" (ICCV 2019)
- Zhou et al., "Unsupervised Learning of Depth and Ego-Motion from Video" (CVPR 2017)
- Jocher et al., "YOLOv8" (Ultralytics, 2023)
