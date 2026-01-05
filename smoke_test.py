#!/usr/bin/env python3
"""
Smoke test script for monocular depth estimation.
Generates synthetic data and runs 1 training step + 1 inference step.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import DepthNet, PoseNet
from datasets import SyntheticDataset
from losses import MonoDepthLoss
from geometry import inverse_warp
from utils import set_seed, count_parameters


def create_synthetic_sequence(output_dir: str, num_frames: int = 5):
    """
    Generate a synthetic 3-frame sequence for testing.
    Creates simple gradient images with slight variations.
    """
    output_path = Path(output_dir) / 'sequence_01'
    output_path.mkdir(parents=True, exist_ok=True)
    
    H, W = 192, 640
    
    for i in range(num_frames):
        # Create a gradient image with some variation
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x, y)
        
        # Add some circles and variation
        r = np.random.uniform(0.1, 0.3)
        cx, cy = np.random.uniform(0.3, 0.7, 2)
        circle = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * r**2))
        
        # Create RGB channels with different patterns
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = ((xx + 0.1 * i / num_frames) * 255).astype(np.uint8)
        img[:, :, 1] = ((yy + circle * 0.5) * 255).astype(np.uint8)
        img[:, :, 2] = ((1 - xx * yy + circle * 0.3) * 255).astype(np.uint8)
        
        # Save
        Image.fromarray(img).save(output_path / f'frame_{i:04d}.png')
    
    print(f"Created synthetic sequence with {num_frames} frames at {output_path}")
    return str(output_path)


def test_geometry():
    """Test geometry operations."""
    print("\n=== Testing Geometry Module ===")
    
    device = torch.device('cpu')
    B, H, W = 2, 64, 128
    
    # Test intrinsics
    from geometry import get_intrinsics, pixel2cam, cam2pixel
    
    K = get_intrinsics(320, 240, 64, 32, device)
    print(f"Intrinsics K shape: {K.shape}")
    
    # Test backprojection
    K_batch = K.unsqueeze(0).expand(B, -1, -1)
    K_inv = torch.inverse(K_batch)
    depth = torch.ones(B, 1, H, W, device=device)
    
    cam_points = pixel2cam(depth, K_inv)
    print(f"Camera points shape: {cam_points.shape}")
    
    # Test projection
    pixel_coords = cam2pixel(cam_points, K_batch)
    print(f"Pixel coords shape: {pixel_coords.shape}")
    
    # Test transform
    from geometry import pose_vec_to_matrix
    
    translation = torch.zeros(B, 3, device=device)
    rotation = torch.zeros(B, 3, device=device)
    T = pose_vec_to_matrix(translation, rotation)
    print(f"Pose matrix shape: {T.shape}")
    
    print("Geometry tests PASSED!")


def test_models():
    """Test model forward passes."""
    print("\n=== Testing Models ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 2, 192, 640
    
    # Test DepthNet
    depth_net = DepthNet(encoder_name='resnet18', pretrained=False).to(device)
    print(f"DepthNet parameters: {count_parameters(depth_net):,}")
    
    x = torch.randn(B, 3, H, W, device=device)
    outputs = depth_net(x)
    
    print(f"DepthNet outputs:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    # Test PoseNet
    pose_net = PoseNet(num_input_images=2).to(device)
    print(f"PoseNet parameters: {count_parameters(pose_net):,}")
    
    target = torch.randn(B, 3, H, W, device=device)
    source = torch.randn(B, 3, H, W, device=device)
    poses = pose_net(target, [source])
    
    print(f"PoseNet outputs: {len(poses)} poses, each {poses[0].shape}")
    
    print("Model tests PASSED!")


def test_losses():
    """Test loss computation."""
    print("\n=== Testing Losses ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W = 2, 192, 640
    
    # Create dummy data
    target = torch.rand(B, 3, H, W, device=device)
    source = torch.rand(B, 3, H, W, device=device)
    
    # Test SSIM
    from losses import compute_ssim_loss
    ssim_loss = compute_ssim_loss(target, source)
    print(f"SSIM loss shape: {ssim_loss.shape}")
    
    # Test photometric
    from losses import compute_photometric_loss
    photo_loss = compute_photometric_loss(target, source)
    print(f"Photometric loss shape: {photo_loss.shape}")
    
    # Test smoothness
    from losses import compute_smoothness_loss
    disp = torch.rand(B, 1, H, W, device=device)
    smooth_loss = compute_smoothness_loss(disp, target)
    print(f"Smoothness loss: {smooth_loss.item():.4f}")
    
    print("Loss tests PASSED!")


def test_training_step():
    """Test one training step."""
    print("\n=== Testing Training Step ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    depth_net = DepthNet(encoder_name='resnet18', pretrained=False).to(device)
    pose_net = PoseNet(num_input_images=2).to(device)
    
    # Create loss
    criterion = MonoDepthLoss(
        ssim_weight=0.85,
        smoothness_weight=0.001,
        use_auto_mask=True,
        scales=[0, 1, 2, 3]
    )
    
    # Create optimizer
    params = list(depth_net.parameters()) + list(pose_net.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    # Create synthetic batch
    B, H, W = 2, 192, 640
    target_img = torch.rand(B, 3, H, W, device=device)
    source_imgs = [
        torch.rand(B, 3, H, W, device=device),
        torch.rand(B, 3, H, W, device=device)
    ]
    
    # Intrinsics
    fx = 0.58 * W
    fy = 0.58 * H
    K = torch.tensor([
        [fx, 0, W/2],
        [0, fy, H/2],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    K = K.unsqueeze(0).expand(B, -1, -1)
    
    # Forward pass
    depth_outputs = depth_net(target_img)
    
    poses = []
    for source_img in source_imgs:
        pose = pose_net(target_img, [source_img])
        poses.extend(pose)
    
    # Compute loss
    losses = criterion(depth_outputs, poses, target_img, source_imgs, K)
    loss = losses['loss']
    
    print(f"Loss values:")
    for key, val in losses.items():
        if key != 'loss':
            print(f"  {key}: {val:.6f}")
    print(f"  Total loss: {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training step PASSED!")


def test_inference():
    """Test inference on a single image."""
    print("\n=== Testing Inference ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    depth_net = DepthNet(encoder_name='resnet18', pretrained=False).to(device)
    depth_net.eval()
    
    # Create test image
    H, W = 192, 640
    test_img = torch.rand(1, 3, H, W, device=device)
    
    # Inference
    with torch.no_grad():
        outputs = depth_net(test_img)
    
    # Check outputs
    disp = outputs[('disp', 0)]
    depth = outputs[('depth', 0)]
    
    print(f"Input shape: {test_img.shape}")
    print(f"Disparity shape: {disp.shape}, range: [{disp.min():.3f}, {disp.max():.3f}]")
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    print("Inference PASSED!")


def test_dataset():
    """Test dataset loading."""
    print("\n=== Testing Dataset ===")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(num_samples=10, height=192, width=640, num_scales=4)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    
    print("Sample contents:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    batch = next(iter(loader))
    print(f"\nBatch target shape: {batch[('color', 0, 0)].shape}")
    
    print("Dataset tests PASSED!")


def run_smoke_test():
    """Run complete smoke test."""
    print("=" * 60)
    print("MONOCULAR DEPTH ESTIMATION - SMOKE TEST")
    print("=" * 60)
    
    set_seed(42)
    
    try:
        # Test each component
        test_geometry()
        test_models()
        test_losses()
        test_dataset()
        test_training_step()
        test_inference()
        
        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)
