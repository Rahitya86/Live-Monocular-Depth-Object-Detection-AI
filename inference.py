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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm

from models import DepthNet
from utils import colorize_depth


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
    window_name: str = "Depth Estimation - Press 'q' to quit"
):
    """Run live display mode."""
    model.eval()
    
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
    
    print("\nControls: 'q'=Quit, 's'=Save, SPACE=Pause")
    
    paused = False
    frame_count = 0
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
            
            combined = np.hstack([frame, depth_colored_bgr])
            
            cv2.putText(combined, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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

