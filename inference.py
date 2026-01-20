#!/usr/bin/env python3
"""
Inference script for monocular depth estimation.

Features:
- Single image, directory of images, or video inference
- Side-by-side colorized visualization
- Live webcam display mode
- Video creation from directory outputs
- Optional raw depth saving (.npy)

Usage:
    # Single image
    python inference.py --input image.jpg --checkpoint checkpoints/best.pth
    
    # Directory of images
    python inference.py --input images/ --checkpoint checkpoints/best.pth
    
    # Video file
    python inference.py --input video.mp4 --checkpoint checkpoints/best.pth
    
    # Live webcam display
    python inference.py --webcam 0 --checkpoint checkpoints/best.pth
    
    # Live display from directory
    python inference.py --input images/ --checkpoint checkpoints/best.pth --live
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
from tqdm import tqdm

from models import DepthNet
from utils import colorize_depth

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("INFO: ultralytics not installed. Using Faster R-CNN instead.")

# Classes to detect
CLASSES_TO_ENHANCE = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
}

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# =============================================================================
# Object Detection Functions
# =============================================================================

def load_yolo_detector(model_name: str = 'yolov8n.pt', device: torch.device = None,
                       confidence: float = 0.35):
    """Load YOLOv8 object detection model."""
    if not YOLO_AVAILABLE:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading YOLOv8 model: {model_name}...")
    try:
        model = YOLO(model_name)
        model.conf = confidence
        print(f"  ✓ YOLOv8 loaded (confidence: {confidence})")
        return model
    except Exception as e:
        print(f"  ERROR loading YOLO: {e}")
        return None


def load_fasterrcnn_detector(device: torch.device = None):
    """Load Faster R-CNN as fallback detector."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Faster R-CNN detector...")
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    detector = detector.to(device)
    detector.eval()
    print("  ✓ Faster R-CNN loaded")
    return detector


def detect_objects_yolo(model, frame: np.ndarray, confidence: float = 0.35) -> List[Dict]:
    """Detect objects using YOLOv8."""
    results = model(frame, verbose=False, conf=confidence)
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = r.names[cls]
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': cls,
                'class_name': class_name
            })
    return detections


def detect_objects_fasterrcnn(model, frame: np.ndarray, device: torch.device,
                               confidence: float = 0.35) -> List[Dict]:
    """Detect objects using Faster R-CNN."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(img_tensor)
    
    detections = []
    pred = predictions[0]
    for i in range(len(pred['boxes'])):
        score = pred['scores'][i].item()
        if score < confidence:
            continue
        box = pred['boxes'][i].cpu().numpy().astype(int)
        label_idx = pred['labels'][i].item()
        if label_idx < len(COCO_CLASSES):
            class_name = COCO_CLASSES[label_idx]
        else:
            class_name = 'unknown'
        detections.append({
            'bbox': (box[0], box[1], box[2], box[3]),
            'confidence': score,
            'class_id': label_idx,
            'class_name': class_name
        })
    return detections


def compute_object_distance(disparity_map: np.ndarray, bbox: Tuple[int, int, int, int],
                            focal_length: float = 721.0, baseline: float = 0.54) -> float:
    """
    Compute accurate distance to object from disparity map.
    
    Uses KITTI camera parameters by default:
    - focal_length: 721 pixels (for 1242x375 resolution, scales with image size)
    - baseline: 0.54m (stereo baseline)
    """
    x1, y1, x2, y2 = bbox
    h, w = disparity_map.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Use central 50% of bounding box for more stable measurement
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    box_w, box_h = x2 - x1, y2 - y1
    inner_x1 = max(x1, cx - box_w // 4)
    inner_y1 = max(y1, cy - box_h // 4)
    inner_x2 = min(x2, cx + box_w // 4)
    inner_y2 = min(y2, cy + box_h // 4)
    
    # Also sample the lower-center region
    lower_y1 = y1 + int(box_h * 0.5)
    lower_y2 = y2
    lower_x1 = max(x1, cx - box_w // 3)
    lower_x2 = min(x2, cx + box_w // 3)
    
    # Get disparity regions
    center_region = disparity_map[inner_y1:inner_y2, inner_x1:inner_x2]
    lower_region = disparity_map[lower_y1:lower_y2, lower_x1:lower_x2]
    
    all_disparities = []
    
    for region in [center_region, lower_region]:
        if region.size == 0:
            continue
        valid_mask = np.isfinite(region) & (region > 0.01)
        valid_vals = region[valid_mask]
        if valid_vals.size > 0:
            all_disparities.extend(valid_vals.flatten())
    
    if len(all_disparities) == 0:
        return 0.0
    
    all_disparities = np.array(all_disparities)
    
    # Use percentile to remove outliers
    if len(all_disparities) > 5:
        p30, p70 = np.percentile(all_disparities, [30, 70])
        filtered = all_disparities[(all_disparities >= p30) & (all_disparities <= p70)]
        if len(filtered) > 0:
            all_disparities = filtered
    
    # Get median disparity value (higher disparity = closer object)
    median_disp = float(np.median(all_disparities))
    
    # Scale focal length based on current image size vs KITTI default
    scaled_focal = focal_length * (w / 1242.0)
    
    # Convert disparity to depth
    pixel_disparity = median_disp * w * 0.3
    
    if pixel_disparity > 0.5:
        depth_meters = (scaled_focal * baseline) / pixel_disparity
        depth_meters = np.clip(depth_meters, 0.5, 80.0)
    else:
        depth_meters = 50.0 / (median_disp + 0.01)
        depth_meters = np.clip(depth_meters, 5.0, 80.0)
    
    return depth_meters


class DepthSmoother:
    """
    Kalman-filter-based temporal smoothing for stable distance measurements.
    Handles object tracking and provides accurate distance even while moving.
    """
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 0.5, 
                 max_history: int = 15, velocity_smoothing: float = 0.7):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_history = max_history
        self.velocity_smoothing = velocity_smoothing
        self.tracks: Dict[str, Dict] = {}
        self.frame_count = 0
    
    def _get_track_key(self, class_name: str, bbox: Tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return f"{class_name}_{cx // 60}_{cy // 60}"
    
    def _find_best_match(self, class_name: str, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        best_key = None
        best_dist = float('inf')
        
        for key, track in self.tracks.items():
            if not key.startswith(class_name):
                continue
            if 'last_bbox' not in track:
                continue
            
            lx1, ly1, lx2, ly2 = track['last_bbox']
            lcx, lcy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
            dist = np.sqrt((cx - lcx)**2 + (cy - lcy)**2)
            
            if dist < 100 and dist < best_dist:
                best_dist = dist
                best_key = key
        
        return best_key
    
    def update(self, class_name: str, distance: float, 
               bbox: Tuple[int, int, int, int]) -> float:
        self.frame_count += 1
        
        if distance <= 0 or distance < 0.3 or distance > 80.0:
            key = self._get_track_key(class_name, bbox)
            if key in self.tracks:
                return self.tracks[key]['distance']
            return 0.0
        
        key = self._find_best_match(class_name, bbox)
        if key is None:
            key = self._get_track_key(class_name, bbox)
        
        if key not in self.tracks:
            self.tracks[key] = {
                'distance': distance,
                'velocity': 0.0,
                'variance': 1.0,
                'history': [distance],
                'last_bbox': bbox,
                'last_frame': self.frame_count
            }
            return distance
        
        track = self.tracks[key]
        dt = 1.0 / 30.0
        predicted_dist = track['distance'] + track['velocity'] * dt
        predicted_var = track['variance'] + self.process_noise
        kalman_gain = predicted_var / (predicted_var + self.measurement_noise)
        new_distance = predicted_dist + kalman_gain * (distance - predicted_dist)
        new_variance = (1 - kalman_gain) * predicted_var
        velocity = (new_distance - track['distance']) / dt
        track['velocity'] = self.velocity_smoothing * velocity + (1 - self.velocity_smoothing) * track['velocity']
        track['velocity'] = np.clip(track['velocity'], -10.0, 10.0)
        track['distance'] = new_distance
        track['variance'] = new_variance
        track['history'].append(distance)
        track['last_bbox'] = bbox
        track['last_frame'] = self.frame_count
        
        if len(track['history']) > self.max_history:
            track['history'].pop(0)
        
        stale_keys = [k for k, v in self.tracks.items() 
                      if self.frame_count - v.get('last_frame', 0) > 30]
        for k in stale_keys:
            del self.tracks[k]
        
        return new_distance


# Global depth smoother for inference.py
_depth_smoother = DepthSmoother()


def draw_detections(frame: np.ndarray, detections: List[Dict], 
                   depth_map: np.ndarray = None, show_distance: bool = True,
                   scaling_factor: float = 10.0) -> np.ndarray:
    """Draw bounding boxes and labels on frame with enhanced visibility."""
    result = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        
        if class_name == 'person':
            color = (0, 255, 0)  # Green
            fill_color = (0, 180, 0)
        elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
            color = (255, 100, 0)  # Blue-orange
            fill_color = (200, 80, 0)
        else:
            color = (0, 255, 255)  # Yellow
            fill_color = (0, 200, 200)
        
        # Create semi-transparent overlay for the detection region
        overlay = result.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
        cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)
        
        # Draw thick border for better visibility
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Add corner markers for enhanced visibility
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        # Top-left corner
        cv2.line(result, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 4)
        cv2.line(result, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 4)
        # Top-right corner
        cv2.line(result, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 4)
        cv2.line(result, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 4)
        # Bottom-left corner
        cv2.line(result, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 4)
        cv2.line(result, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 4)
        # Bottom-right corner
        cv2.line(result, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 4)
        cv2.line(result, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 4)
        
        distance_str = ""
        if show_distance and depth_map is not None:
            # Use accurate distance calculation with KITTI camera parameters
            distance = compute_object_distance(depth_map, (x1, y1, x2, y2))
            # Apply temporal smoothing for stable measurements
            distance = _depth_smoother.update(class_name, distance, (x1, y1, x2, y2))
            if distance > 0:
                distance_str = f": {distance:.1f}m"
        
        label = f"{class_name.capitalize()}{distance_str}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Label background with padding
        label_y = max(0, y1 - th - 12)
        cv2.rectangle(result, (x1, label_y), (x1 + tw + 12, y1), color, -1)
        cv2.rectangle(result, (x1, label_y), (x1 + tw + 12, y1), (255, 255, 255), 2)
        cv2.putText(result, label, (x1 + 6, y1 - 6), font, font_scale, (255, 255, 255), thickness)
        
        # Add depth indicator bar at bottom of box (inverse: closer = longer bar)
        if show_distance and depth_map is not None and distance_str:
            bar_height = 8
            bar_y = y2 - bar_height - 4
            dist_val = float(distance_str.replace(': ', '').replace('m', ''))
            # Inverse relationship: closer objects get longer bar
            bar_ratio = max(0.1, 1.0 - (dist_val / 50.0))
            bar_width = int(max(10, (x2 - x1 - 4) * bar_ratio))
            cv2.rectangle(result, (x1 + 2, bar_y), (x1 + 2 + bar_width, y2 - 4), (0, 200, 255), -1)
            cv2.rectangle(result, (x1 + 2, bar_y), (x2 - 2, y2 - 4), (255, 255, 255), 1)
    
    return result


# =============================================================================
# Image Processing Utilities
# =============================================================================

def load_image(path: str, height: int = 192, width: int = 640) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """Load and preprocess image."""
    img = Image.open(path).convert('RGB')
    original = np.array(img)
    original_size = (img.height, img.width)
    
    img_resized = img.resize((width, height), Image.LANCZOS)
    
    # Normalize to ImageNet stats
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized).unsqueeze(0)
    
    return tensor, original, original_size


def save_depth_output(
    depth: torch.Tensor,
    output_path: str,
    original_size: Optional[Tuple[int, int]] = None,
    colorize: bool = True,
    save_numpy: bool = False,
    min_depth: float = 0.1,
    max_depth: float = 100.0
):
    """Save depth output as colorized image and/or numpy array."""
    # Convert to numpy
    if depth.dim() == 4:
        depth = depth.squeeze(0)
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    depth_np = depth.cpu().numpy()
    
    # Resize to original size if needed
    if original_size is not None:
        h, w = original_size
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Save colorized
    if colorize:
        # Normalize for visualization
        depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        cmap = plt.cm.magma
        depth_colored = cmap(depth_normalized)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        Image.fromarray(depth_colored).save(output_path)
    
    # Save numpy array
    if save_numpy:
        npy_path = output_path.replace('.png', '.npy').replace('.jpg', '.npy')
        np.save(npy_path, depth_np)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(
    checkpoint_path: str,
    encoder_name: str = 'resnet18',
    device: torch.device = torch.device('cpu'),
    lightweight: bool = True
) -> DepthNet:
    """Load trained model from checkpoint."""
    print(f"Creating DepthNet with {encoder_name} encoder...")
    
    model = DepthNet(
        encoder_name=encoder_name,
        pretrained=False,
        scales=[0, 1, 2, 3],
        min_depth=0.1,
        max_depth=100.0,
        use_attention=False,
        lightweight=lightweight
    )
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats (use strict=False for multi-scale compatibility)
        if 'depth_net_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['depth_net_state_dict'], strict=False)
        elif 'depth_net' in checkpoint:
            model.load_state_dict(checkpoint['depth_net'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("  Checkpoint loaded successfully!")
    else:
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        print("  Using random weights (results will be meaningless)")
    
    model = model.to(device)
    model.eval()
    
    return model


# =============================================================================
# Inference Functions
# =============================================================================

@torch.no_grad()
def predict_depth(
    model: DepthNet,
    image: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Predict depth from image tensor."""
    image = image.to(device)
    outputs = model(image)
    depth = outputs[('depth', 0)]
    return depth


def process_single_image(
    model: DepthNet,
    image_path: str,
    output_dir: str,
    device: torch.device,
    height: int = 192,
    width: int = 640,
    save_numpy: bool = False,
    resize_to_original: bool = True
):
    """Process a single image."""
    # Load image
    tensor, original, original_size = load_image(image_path, height, width)
    
    # Predict depth
    depth = predict_depth(model, tensor, device)
    
    # Save output
    filename = Path(image_path).stem + '_depth.png'
    output_path = os.path.join(output_dir, filename)
    
    save_depth_output(
        depth,
        output_path,
        original_size if resize_to_original else None,
        colorize=True,
        save_numpy=save_numpy
    )
    
    print(f"  Saved: {output_path}")


def process_directory(
    model: DepthNet,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    height: int = 192,
    width: int = 640,
    save_numpy: bool = False,
    create_video: bool = True,
    video_fps: float = 10.0
):
    """Process all images in a directory."""
    # Find images
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(extensions)
    ])
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Colormap for depth
    cmap = plt.cm.magma
    video_frames = []
    
    # Process each image
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, filename)
        
        # Load and predict
        tensor, original, original_size = load_image(image_path, height, width)
        depth = predict_depth(model, tensor, device)
        
        # Convert depth to visualization
        depth_np = depth.squeeze().cpu().numpy()
        depth_resized = cv2.resize(depth_np, (original_size[1], original_size[0]))
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
        depth_colored = cmap(depth_normalized)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        
        # Save colorized depth
        stem = Path(filename).stem
        output_path = os.path.join(output_dir, f"{stem}_depth.png")
        Image.fromarray(depth_colored).save(output_path)
        print(f"  Saved colorized depth: {output_path}")
        
        # Collect frames for video
        if create_video:
            combined = np.hstack([original, depth_colored])
            video_frames.append(combined)
        
        if save_numpy:
            np.save(output_path.replace('.png', '.npy'), depth_resized)
    
    # Create video
    if create_video and video_frames:
        video_path = os.path.join(output_dir, 'depth_video.mp4')
        print(f"\nCreating combined video: {video_path}")
        
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))
        
        for frame in tqdm(video_frames, desc="Writing video"):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"  Video saved: {video_path}")
        print(f"  Resolution: {w}x{h}")
        print(f"  FPS: {video_fps}")
        print(f"  Frames: {len(video_frames)}")


def run_live_display(
    model: DepthNet,
    source: str,
    device: torch.device,
    height: int = 192,
    width: int = 640,
    window_name: str = "Depth Estimation - Press 'q' to quit",
    detector=None,
    use_yolo: bool = True,
    detector_confidence: float = 0.35
):
    """Run live display mode with object detection."""
    model.eval()
    
    # Load detector if not provided
    if detector is None:
        if YOLO_AVAILABLE:
            detector = load_yolo_detector(confidence=detector_confidence)
            use_yolo = True
        else:
            detector = load_fasterrcnn_detector(device)
            use_yolo = False
    
    # Determine source type
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        mode = 'webcam'
        print(f"Opening webcam {source}...")
    elif os.path.isdir(source):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = sorted([
            os.path.join(source, f) for f in os.listdir(source)
            if f.lower().endswith(extensions)
        ])
        if not image_files:
            print(f"No images found in {source}")
            return
        mode = 'directory'
        frame_idx = 0
        print(f"Found {len(image_files)} images")
    else:
        cap = cv2.VideoCapture(source)
        mode = 'video'
        print(f"Opening video: {source}")
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)
    
    print("\nControls: 'q'=Quit, 's'=Save, SPACE=Pause, 'd'=Toggle detections")
    
    paused = False
    frame_count = 0
    show_detections = True
    cmap = plt.cm.magma
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        while True:
            if mode in ['webcam', 'video']:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        if mode == 'video':
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break
            else:
                if not paused:
                    if frame_idx >= len(image_files):
                        frame_idx = 0
                    frame = cv2.imread(image_files[frame_idx])
                    frame_idx += 1
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = frame_rgb.shape[:2]
            
            # Object detection
            detections = []
            if show_detections and detector is not None:
                try:
                    if use_yolo and YOLO_AVAILABLE:
                        detections = detect_objects_yolo(detector, frame, detector_confidence)
                    else:
                        detections = detect_objects_fasterrcnn(detector, frame, device, detector_confidence)
                    detections = [d for d in detections if d['class_name'] in CLASSES_TO_ENHANCE]
                except Exception as e:
                    detections = []
            
            input_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                disp = outputs[('disp', 0)]
            
            disp_np = disp.squeeze().cpu().numpy()
            disp_resized = cv2.resize(disp_np, (w_orig, h_orig))
            disp_normalized = (disp_resized - disp_resized.min()) / (disp_resized.max() - disp_resized.min() + 1e-8)
            
            depth_colored = cmap(disp_normalized)[:, :, :3]
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            
            # Draw detections on depth map
            if show_detections and detections:
                depth_with_boxes = draw_detections(depth_colored_bgr, detections, disp_resized, show_distance=True)
            else:
                depth_with_boxes = depth_colored_bgr
            
            # Show depth with detections only
            combined = depth_with_boxes
            
            # Add overlay
            cv2.rectangle(combined, (0, 0), (350, 60), (0, 0, 0), -1)
            cv2.putText(combined, f"Frame: {frame_count} | Objects: {len(detections)}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, "Q:quit S:save SPACE:pause D:detections", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            if paused:
                cv2.putText(combined, "PAUSED", (combined.shape[1]//2 - 50, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(window_name, combined)
            frame_count += 1
            
            key = cv2.waitKey(1 if mode == 'webcam' else 30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"outputs/live_frame_{frame_count:06d}.png"
                os.makedirs("outputs", exist_ok=True)
                cv2.imwrite(save_path, combined)
                print(f"Saved: {save_path}")
            elif key == ord(' '):
                paused = not paused
            elif key == ord('d'):
                show_detections = not show_detections
                print(f"Detections: {'ON' if show_detections else 'OFF'}")
    
    finally:
        if mode in ['webcam', 'video']:
            cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Monocular depth estimation inference')
    parser.add_argument('--input', type=str, default=None, help='Input image/directory/video')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth', help='Model checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet18', help='Encoder architecture')
    parser.add_argument('--height', type=int, default=192, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--save-numpy', action='store_true', help='Save raw depth arrays')
    parser.add_argument('--live', action='store_true', help='Live display mode')
    parser.add_argument('--webcam', type=int, default=None, help='Webcam device ID')
    parser.add_argument('--fps', type=float, default=10.0, help='Video output FPS')
    parser.add_argument('--no-video', action='store_true', help='Disable video output')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, args.encoder, device, lightweight=True)
    
    # Webcam mode
    if args.webcam is not None:
        run_live_display(model, str(args.webcam), device, args.height, args.width)
        return
    
    # Live display mode
    if args.live and args.input:
        run_live_display(model, args.input, device, args.height, args.width)
        return
    
    # Require input for other modes
    if args.input is None:
        print("ERROR: --input is required (or use --webcam)")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}")
    
    if os.path.isdir(args.input):
        process_directory(
            model, args.input, args.output, device,
            args.height, args.width, args.save_numpy,
            create_video=not args.no_video, video_fps=args.fps
        )
    elif os.path.isfile(args.input):
        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            run_live_display(model, args.input, device, args.height, args.width)
        else:
            process_single_image(
                model, args.input, args.output, device,
                args.height, args.width, args.save_numpy
            )
    else:
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)
    
    print("\nInference complete!")


if __name__ == '__main__':
    main()

