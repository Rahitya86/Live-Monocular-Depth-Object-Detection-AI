#!/usr/bin/env python3
"""
Live Monocular Depth Estimation with Object Detection.

Usage:
    python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth
    python inference_live.py --input video.mp4 --checkpoint checkpoints/best.pth
    python inference_live.py --input images/ --checkpoint checkpoints/best.pth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from models import DepthNet

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("INFO: ultralytics not installed. Using Faster R-CNN instead.")

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

COLORMAPS = {
    'accurate': None, 'magma': cv2.COLORMAP_MAGMA, 'viridis': cv2.COLORMAP_VIRIDIS,
    'plasma': cv2.COLORMAP_PLASMA, 'inferno': cv2.COLORMAP_INFERNO,
    'jet': cv2.COLORMAP_JET, 'turbo': cv2.COLORMAP_TURBO, 'hot': cv2.COLORMAP_HOT,
}


def load_depth_model(checkpoint_path: str, encoder_name: str = 'resnet18', 
                     device: torch.device = None) -> DepthNet:
    """Load trained depth estimation model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading DepthNet ({encoder_name}) to {device}...")
    model = DepthNet(
        encoder_name=encoder_name, pretrained=False, scales=[0, 1, 2, 3],
        min_depth=0.1, max_depth=100.0, use_attention=False, lightweight=True
    )
    model = model.to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'depth_net_state_dict' in ck:
            model.load_state_dict(ck['depth_net_state_dict'], strict=False)
        elif 'depth_net' in ck:
            model.load_state_dict(ck['depth_net'], strict=False)
        elif 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'], strict=False)
        elif 'state_dict' in ck:
            model.load_state_dict(ck['state_dict'], strict=False)
        else:
            model.load_state_dict(ck, strict=False)
        print("  ✓ Depth model loaded!")
    else:
        print("  WARNING: No checkpoint found — using random weights")
    
    model.eval()
    return model


def load_yolo_detector(model_name: str = 'yolov8n.pt', device: torch.device = None,
                       confidence: float = 0.35) -> Optional['YOLO']:
    """Load YOLOv8 object detection model."""
    if not YOLO_AVAILABLE:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading YOLOv8 model: {model_name}...")
    try:
        model = YOLO(model_name)
        model.to(device)
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
    detector.to(device)
    detector.eval()
    print("  ✓ Faster R-CNN loaded")
    return detector


@torch.no_grad()
def detect_objects_yolo(yolo_model: 'YOLO', frame: np.ndarray, 
                        confidence: float = 0.35) -> List[Dict]:
    """Detect objects using YOLOv8."""
    results = yolo_model(frame, conf=confidence, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names.get(cls_id, 'object')
            detections.append({
                'class_name': class_name, 'confidence': conf, 'bbox': (x1, y1, x2, y2)
            })
    return detections


@torch.no_grad()
def detect_objects_fasterrcnn(detector, frame: np.ndarray, device: torch.device,
                              confidence: float = 0.5) -> List[Dict]:
    """Detect objects using Faster R-CNN."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = T.ToTensor()(Image.fromarray(frame_rgb)).to(device)
    results = detector([tensor])[0]
    
    detections = []
    for box, label, score in zip(results['boxes'], results['labels'], results['scores']):
        if score < confidence:
            continue
        cls_id = int(label)
        class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else 'object'
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        detections.append({
            'class_name': class_name, 'confidence': float(score), 'bbox': (x1, y1, x2, y2)
        })
    return detections


def compute_object_distance(disparity_map: np.ndarray, bbox: Tuple[int, int, int, int], 
                            focal_length: float = 721.0, baseline: float = 0.54) -> float:
    """
    Compute accurate distance to object from disparity map.
    
    Uses KITTI camera parameters by default:
    - focal_length: 721 pixels (for 1242x375 resolution, scales with image size)
    - baseline: 0.54m (stereo baseline)
    
    For monocular depth, we use the disparity-to-depth conversion:
    depth = (focal_length * baseline) / disparity
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
    
    # Also sample the lower-center region (usually more reliable for objects on ground)
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
    
    # The model outputs normalized disparity in range ~[0, 1]
    # Higher disparity = closer, lower disparity = farther
    # We need to map this to real-world distance
    
    # Scale focal length based on current image size vs KITTI default (1242 width)
    scaled_focal = focal_length * (w / 1242.0)
    
    # Convert disparity to depth using stereo formula
    # For normalized disparity, we scale by image width to get pixel disparity
    pixel_disparity = median_disp * w * 0.3  # Scale factor for model output
    
    if pixel_disparity > 0.5:
        depth_meters = (scaled_focal * baseline) / pixel_disparity
        depth_meters = np.clip(depth_meters, 0.5, 80.0)
    else:
        # Use inverse relationship for very small disparities
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
        
        # State: {key: {'distance': float, 'velocity': float, 'variance': float, 'history': list, 'last_bbox': tuple}}
        self.tracks: Dict[str, Dict] = {}
        self.frame_count = 0
    
    def _get_track_key(self, class_name: str, bbox: Tuple[int, int, int, int]) -> str:
        """Generate tracking key based on position."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Use grid-based tracking for object association
        grid_x, grid_y = cx // 60, cy // 60
        return f"{class_name}_{grid_x}_{grid_y}"
    
    def _find_best_match(self, class_name: str, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """Find best matching existing track for this detection."""
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
            
            # Match if within 100 pixels
            if dist < 100 and dist < best_dist:
                best_dist = dist
                best_key = key
        
        return best_key
    
    def update(self, class_name: str, distance: float, 
               bbox: Tuple[int, int, int, int]) -> float:
        """
        Update distance estimate using Kalman filter approach.
        Returns smoothed, accurate distance measurement.
        """
        self.frame_count += 1
        
        # Validate measurement
        if distance <= 0 or distance < 0.3 or distance > 80.0:
            key = self._get_track_key(class_name, bbox)
            if key in self.tracks:
                return self.tracks[key]['distance']
            return 0.0
        
        # Try to find existing track
        key = self._find_best_match(class_name, bbox)
        if key is None:
            key = self._get_track_key(class_name, bbox)
        
        if key not in self.tracks:
            # Initialize new track
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
        
        # Predict step (constant velocity model)
        dt = 1.0 / 30.0  # Assume 30 FPS
        predicted_dist = track['distance'] + track['velocity'] * dt
        predicted_var = track['variance'] + self.process_noise
        
        # Update step (Kalman gain)
        kalman_gain = predicted_var / (predicted_var + self.measurement_noise)
        
        # Update distance estimate
        new_distance = predicted_dist + kalman_gain * (distance - predicted_dist)
        new_variance = (1 - kalman_gain) * predicted_var
        
        # Update velocity estimate
        velocity = (new_distance - track['distance']) / dt
        track['velocity'] = self.velocity_smoothing * velocity + (1 - self.velocity_smoothing) * track['velocity']
        
        # Limit velocity to reasonable range (-10 to 10 m/s)
        track['velocity'] = np.clip(track['velocity'], -10.0, 10.0)
        
        # Update track
        track['distance'] = new_distance
        track['variance'] = new_variance
        track['history'].append(distance)
        track['last_bbox'] = bbox
        track['last_frame'] = self.frame_count
        
        if len(track['history']) > self.max_history:
            track['history'].pop(0)
        
        # Clean up old tracks
        self._cleanup_old_tracks()
        
        return new_distance
    
    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been updated recently."""
        stale_keys = [k for k, v in self.tracks.items() 
                      if self.frame_count - v.get('last_frame', 0) > 30]
        for key in stale_keys:
            del self.tracks[key]


@torch.no_grad()
def predict_depth(model: DepthNet, frame: np.ndarray, device: torch.device,
                  height: int = 192, width: int = 640) -> np.ndarray:
    """Predict depth from BGR frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(Image.fromarray(frame_resized)).unsqueeze(0).to(device)
    
    outputs = model(tensor)
    
    if ('disp', 0) in outputs:
        disp = outputs[('disp', 0)]
    elif ('depth', 0) in outputs:
        disp = 1.0 / (outputs[('depth', 0)] + 1e-8)
    else:
        disp = list(outputs.values())[0]
    
    depth_np = disp.squeeze().cpu().numpy()
    H, W = frame.shape[:2]
    return cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)


def draw_detections(frame: np.ndarray, detections: List[Dict], 
                   depth_map: np.ndarray = None, smoother: DepthSmoother = None,
                   show_distance: bool = True) -> np.ndarray:
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
            distance = compute_object_distance(depth_map, (x1, y1, x2, y2))
            if smoother and distance > 0:
                distance = smoother.update(class_name, distance, (x1, y1, x2, y2))
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


class LiveDepthProcessor:
    """Live depth estimation with object detection."""
    
    def __init__(self, depth_model: DepthNet, detector, device: torch.device,
                 height: int = 192, width: int = 640, colormap: str = 'accurate',
                 detector_confidence: float = 0.35, draw_boxes: bool = True,
                 use_yolo: bool = True, show_distance: bool = True,
                 scaling_factor: float = 10.0):
        self.depth_model = depth_model
        self.detector = detector
        self.device = device
        self.height = height
        self.width = width
        self.colormap = colormap
        self.detector_confidence = detector_confidence
        self.draw_boxes = draw_boxes
        self.use_yolo = use_yolo and YOLO_AVAILABLE and detector is not None
        self.show_distance = show_distance
        self.scaling_factor = scaling_factor
        
        self.frame_count = 0
        self.fps = 0.0
        self.paused = False
        self.frame_times = []
        self.depth_smoother = DepthSmoother()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Process a single frame."""
        t0 = time.time()
        
        # Object detection
        if self.detector is not None and self.draw_boxes:
            try:
                if self.use_yolo:
                    detections = detect_objects_yolo(self.detector, frame, self.detector_confidence)
                else:
                    detections = detect_objects_fasterrcnn(self.detector, frame, self.device, self.detector_confidence)
                detections = [d for d in detections if d['class_name'] in CLASSES_TO_ENHANCE]
            except:
                detections = []
        else:
            detections = []
        
        # Depth prediction - get raw disparity
        disparity_map = predict_depth(self.depth_model, frame, self.device, self.height, self.width)
        
        # Get disparity stats for normalization
        disp = disparity_map.astype(np.float32)
        disp_min, disp_max = disp.min(), disp.max()
        if disp_max - disp_min > 1e-8:
            disp_norm = (disp - disp_min) / (disp_max - disp_min)
        else:
            disp_norm = np.zeros_like(disp)
        
        # Colorize depth using normalized disparity
        if self.colormap in ['accurate', 'magma']:
            cmap = plt.cm.magma
            depth_rgb = cmap(disp_norm)[:, :, :3]
            depth_colored = (depth_rgb * 255).astype(np.uint8)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
        else:
            depth_uint8 = (disp_norm * 255).astype(np.uint8)
            cmap_cv = COLORMAPS.get(self.colormap, cv2.COLORMAP_MAGMA)
            if cmap_cv is not None:
                depth_colored = cv2.applyColorMap(depth_uint8, cmap_cv)
            else:
                depth_colored = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)
        
        # Draw detections on depth map - pass RAW disparity for distance calculation
        if self.draw_boxes and detections:
            depth_with_boxes = draw_detections(
                depth_colored, detections, disparity_map if self.show_distance else None,
                self.depth_smoother, self.show_distance
            )
        else:
            depth_with_boxes = depth_colored
        
        # Show only depth with detections
        combined = depth_with_boxes
        
        # Update FPS
        elapsed = time.time() - t0
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        self.fps = 1.0 / (sum(self.frame_times) / len(self.frame_times) + 1e-9)
        self.frame_count += 1
        
        return combined, disparity_map, detections
    
    def add_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Add overlay with stats."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Objects: {len(detections)} | Mode: {self.colormap.upper()}", 
                   (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "SPACE:pause S:save C:colormap Q:quit", (w - 380, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return frame
    
    def cycle_colormap(self):
        """Cycle through colormaps."""
        cmap_names = list(COLORMAPS.keys())
        idx = cmap_names.index(self.colormap) if self.colormap in cmap_names else 0
        self.colormap = cmap_names[(idx + 1) % len(cmap_names)]
        print(f"  Colormap: {self.colormap}")


def run_live_webcam(processor: LiveDepthProcessor, webcam_id: int = 0):
    """Run live depth estimation from webcam."""
    print(f"\nOpening webcam {webcam_id}...")
    cap = cv2.VideoCapture(webcam_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam {webcam_id}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = "Live Depth Estimation - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 500)
    
    print("\nControls: SPACE=pause, S=save, C=colormap, Q=quit\n")
    
    try:
        while True:
            if not processor.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                combined, _, detections = processor.process_frame(frame)
                combined = processor.add_overlay(combined, detections)
                cv2.imshow(window_name, combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                processor.paused = not processor.paused
                print("  PAUSED" if processor.paused else "  RESUMED")
            elif key == ord('c'):
                processor.cycle_colormap()
            elif key == ord('s'):
                path = f"frame_{processor.frame_count:06d}.png"
                cv2.imwrite(path, combined)
                print(f"  Saved: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {processor.frame_count} frames")


def run_live_video(processor: LiveDepthProcessor, video_path: str, slowdown: int = 1):
    """Run depth estimation on video file."""
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        return
    
    window_name = "Depth Estimation - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 500)
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int((1000 / fps) * slowdown)
    
    print(f"Video FPS: {fps:.1f}, Delay: {delay}ms\n")
    
    try:
        while True:
            if not processor.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                combined, _, detections = processor.process_frame(frame)
                combined = processor.add_overlay(combined, detections)
                cv2.imshow(window_name, combined)
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                processor.paused = not processor.paused
            elif key == ord('c'):
                processor.cycle_colormap()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {processor.frame_count} frames")


def run_live_images(processor: LiveDepthProcessor, images: List, slowdown: int = 1, loop: bool = False):
    """Run depth estimation on image sequence."""
    if not images:
        print("No images found")
        return
    
    print(f"\nProcessing {len(images)} images...")
    
    window_name = "Depth Estimation - Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 500)
    
    delay = 30 * slowdown
    idx = 0
    
    try:
        while True:
            if not processor.paused:
                img_path = images[idx] if isinstance(images[idx], str) else images[idx]['path']
                
                frame = cv2.imread(img_path)
                if frame is None:
                    idx = (idx + 1) % len(images)
                    continue
                
                combined, _, detections = processor.process_frame(frame)
                combined = processor.add_overlay(combined, detections)
                cv2.imshow(window_name, combined)
                
                idx += 1
                if idx >= len(images):
                    if loop:
                        idx = 0
                    else:
                        break
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                processor.paused = not processor.paused
            elif key == ord('c'):
                processor.cycle_colormap()
    finally:
        cv2.destroyAllWindows()
        print(f"Processed {processor.frame_count} frames")


def gather_images(directory: str) -> List[str]:
    """Gather images from directory."""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return sorted([os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith(extensions)])


def main():
    parser = argparse.ArgumentParser(description='Live Monocular Depth Estimation')
    
    parser.add_argument('--input', type=str, help='Input image/video file or directory')
    parser.add_argument('--webcam', type=int, help='Webcam ID')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--yolo', type=str, default='yolov8n.pt')
    parser.add_argument('--height', type=int, default=192)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--colormap', type=str, default='accurate', choices=list(COLORMAPS.keys()))
    parser.add_argument('--detector-thresh', type=float, default=0.35)
    parser.add_argument('--no-boxes', action='store_true')
    parser.add_argument('--no-distance', action='store_true')
    parser.add_argument('--slowdown', type=int, default=1)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--scaling-factor', type=float, default=10.0)
    
    args = parser.parse_args()
    
    if args.input is None and args.webcam is None:
        parser.print_help()
        print("\nERROR: Specify --input or --webcam")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint and not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(base_path, args.checkpoint)
    
    depth_model = load_depth_model(args.checkpoint, args.encoder, device)
    
    if YOLO_AVAILABLE:
        detector = load_yolo_detector(args.yolo, device, args.detector_thresh)
        use_yolo = detector is not None
    else:
        detector = load_fasterrcnn_detector(device)
        use_yolo = False
    
    processor = LiveDepthProcessor(
        depth_model=depth_model, detector=detector, device=device,
        height=args.height, width=args.width, colormap=args.colormap,
        detector_confidence=args.detector_thresh, draw_boxes=not args.no_boxes,
        use_yolo=use_yolo, show_distance=not args.no_distance,
        scaling_factor=args.scaling_factor
    )
    
    print(f"\nSettings: {args.width}x{args.height}, {args.colormap}, "
          f"{'YOLOv8' if use_yolo else 'Faster R-CNN'}")
    
    if args.webcam is not None:
        run_live_webcam(processor, args.webcam)
    elif args.input:
        if os.path.isdir(args.input):
            images = gather_images(args.input)
            run_live_images(processor, images, args.slowdown, args.loop)
        elif os.path.isfile(args.input):
            ext = os.path.splitext(args.input)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                run_live_video(processor, args.input, args.slowdown)
            else:
                run_live_images(processor, [args.input], args.slowdown, args.loop)
        else:
            print(f"ERROR: Input not found: {args.input}")
            sys.exit(1)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()
