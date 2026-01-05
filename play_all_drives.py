#!/usr/bin/env python3
"""
Play depth predictions from all KITTI drives in a single live window.
Combines all drives and shows them sequentially.
"""

import os
import sys
import glob
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def load_model(checkpoint_path, encoder_name='resnet18', device='cuda'):
    """Load the depth model from checkpoint."""
    from models.depth_net import DepthNet
    
    model = DepthNet(encoder_name=encoder_name, pretrained=False, lightweight=True)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    return model


def get_all_images_from_drives(base_path):
    """Find all KITTI drives and return sorted list of all images with drive info."""
    all_images = []
    
    # Find all drive directories
    drive_pattern = os.path.join(base_path, 'datasets', '*_drive_*_sync')
    drives = sorted(glob.glob(drive_pattern))
    
    if not drives:
        # Try direct path
        drive_pattern = os.path.join(base_path, '*_drive_*_sync')
        drives = sorted(glob.glob(drive_pattern))
    
    print(f"\nFound {len(drives)} KITTI drives:")
    
    for drive_path in drives:
        drive_name = os.path.basename(drive_path)
        image_dir = os.path.join(drive_path, 'image_02', 'data')
        
        if os.path.exists(image_dir):
            images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            if not images:
                images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
            
            print(f"  - {drive_name}: {len(images)} images")
            
            for img_path in images:
                all_images.append({
                    'path': img_path,
                    'drive': drive_name,
                    'filename': os.path.basename(img_path)
                })
    
    print(f"\nTotal images: {len(all_images)}")
    return all_images


def process_image(image_path, model, device, target_size=(640, 192)):
    """Process single image through the model."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    
    # Transform for model
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        disp = outputs[('disp', 0)]
    
    # Convert to numpy
    disp_np = disp.squeeze().cpu().numpy()
    
    # Normalize for visualization
    disp_np = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)
    
    # Apply colormap
    disp_colored = cv2.applyColorMap((disp_np * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # Resize disparity to match original image width for side-by-side view
    original_np = np.array(img)
    original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
    
    # Resize both to same height
    h = 375
    w_orig = int(original_bgr.shape[1] * h / original_bgr.shape[0])
    
    original_resized = cv2.resize(original_bgr, (w_orig, h))
    disp_resized = cv2.resize(disp_colored, (w_orig, h))
    
    # Combine side by side
    combined = np.hstack([original_resized, disp_resized])
    
    return combined


def run_live_display(base_path, checkpoint_path, encoder_name='resnet18', fps=15, loop=False):
    """Run live display of all KITTI drives."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, encoder_name, device)
    print("Model loaded successfully!")
    
    # Get all images from all drives
    all_images = get_all_images_from_drives(base_path)
    
    if not all_images:
        print("No images found!")
        return
    
    # Create window
    cv2.namedWindow('Depth Estimation - All KITTI Drives', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth Estimation - All KITTI Drives', 1600, 450)
    
    delay = int(1000 / fps)
    current_drive = None
    
    print(f"\nPlaying at {fps} FPS (press 'q' to quit, 'p' to pause, SPACE to resume)")
    print("-" * 60)
    
    paused = False
    idx = 0
    
    while True:
        if not paused:
            img_info = all_images[idx]
            
            # Print drive change
            if img_info['drive'] != current_drive:
                current_drive = img_info['drive']
                print(f"\n>>> Now playing: {current_drive}")
            
            # Process and display
            try:
                combined = process_image(img_info['path'], model, device)
                
                # Add text overlay
                text = f"Drive: {img_info['drive']} | Frame: {img_info['filename']} | {idx+1}/{len(all_images)}"
                cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Depth Estimation - All KITTI Drives', combined)
                
            except Exception as e:
                print(f"Error processing {img_info['path']}: {e}")
            
            idx += 1
            if idx >= len(all_images):
                if loop:
                    idx = 0
                    print("\n>>> Looping back to start...")
                else:
                    print("\n>>> Finished all images!")
                    break
        
        # Handle key presses
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            paused = True
            print("Paused - press SPACE to resume")
        elif key == ord(' '):
            paused = False
            print("Resumed")
    
    cv2.destroyAllWindows()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Play depth estimation on all KITTI drives')
    parser.add_argument('--base_path', type=str, default='/home/rahita/monodepth-starter',
                        help='Base path containing datasets folder')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet18',
                        help='Encoder type (resnet18, resnet34, resnet50)')
    parser.add_argument('--fps', type=int, default=15,
                        help='Playback frames per second')
    parser.add_argument('--loop', action='store_true',
                        help='Loop playback continuously')
    
    args = parser.parse_args()
    
    # Make checkpoint path absolute if needed
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(args.base_path, args.checkpoint)
    
    run_live_display(
        base_path=args.base_path,
        checkpoint_path=args.checkpoint,
        encoder_name=args.encoder,
        fps=args.fps,
        loop=args.loop
    )


if __name__ == '__main__':
    main()
