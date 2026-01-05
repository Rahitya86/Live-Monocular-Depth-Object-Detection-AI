"""
Differentiable inverse warping for view synthesis.
Core operation for self-supervised depth learning.
"""

import torch
import torch.nn.functional as F

from .camera import pixel2cam, cam2pixel, normalize_coords


def inverse_warp(
    source_img: torch.Tensor,
    depth: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    K_inv: torch.Tensor,
    padding_mode: str = 'zeros'
) -> tuple:
    """
    Inverse warp a source image to the target view using depth and pose.
    
    Given depth map of target frame, camera intrinsics, and relative pose
    from target to source, warp source image pixels to reconstruct target view.
    
    Args:
        source_img: (B, 3, H, W) source image to warp
        depth: (B, 1, H_d, W_d) depth map of target frame (may differ from H, W)
        pose: (B, 4, 4) relative pose from target to source frame
        K: (B, 3, 3) camera intrinsics
        K_inv: (B, 3, 3) inverse camera intrinsics
        padding_mode: 'zeros' or 'border' for out-of-bounds pixels
        
    Returns:
        warped_img: (B, 3, H, W) source image warped to target view
        valid_mask: (B, 1, H, W) mask of valid projected pixels
    """
    B, _, H, W = source_img.shape
    _, _, H_d, W_d = depth.shape
    device = source_img.device
    
    # Resize depth to match source image if needed
    if H_d != H or W_d != W:
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)
    
    # Step 1: Backproject target pixels to 3D using depth
    # cam_points: (B, 3, H, W) - 3D points in target camera frame
    cam_points = pixel2cam(depth, K_inv)
    
    # Step 2: Transform 3D points to source camera frame
    # Reshape to (B, 3, H*W) for matrix multiplication
    cam_flat = cam_points.view(B, 3, -1)
    
    # Add homogeneous coordinate
    ones = torch.ones(B, 1, H * W, device=device)
    cam_homog = torch.cat([cam_flat, ones], dim=1)  # (B, 4, H*W)
    
    # Apply pose transformation: source_points = pose @ target_points
    source_cam = torch.bmm(pose[:, :3, :], cam_homog)  # (B, 3, H*W)
    source_cam = source_cam.view(B, 3, H, W)
    
    # Step 3: Project to source image plane
    # Extract depth in source frame for validity check
    source_depth = source_cam[:, 2:3, :, :]  # (B, 1, H, W)
    
    # Project to pixel coordinates
    pixel_coords = cam2pixel(source_cam, K)  # (B, 2, H, W)
    
    # Step 4: Normalize coordinates for grid_sample
    grid = normalize_coords(pixel_coords, H, W)  # (B, H, W, 2)
    
    # Step 5: Sample from source image
    warped_img = F.grid_sample(
        source_img,
        grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True
    )
    
    # Step 6: Create validity mask
    # Valid if: within image bounds and positive depth
    x_valid = (pixel_coords[:, 0, :, :] >= 0) & (pixel_coords[:, 0, :, :] < W)
    y_valid = (pixel_coords[:, 1, :, :] >= 0) & (pixel_coords[:, 1, :, :] < H)
    depth_valid = source_depth[:, 0, :, :] > 0
    
    valid_mask = (x_valid & y_valid & depth_valid).unsqueeze(1).float()
    
    return warped_img, valid_mask


def compute_projection_coords(
    depth: torch.Tensor,
    pose: torch.Tensor,
    K: torch.Tensor,
    K_inv: torch.Tensor
) -> tuple:
    """
    Compute projected pixel coordinates without warping.
    Useful for computing optical flow from depth + pose.
    
    Args:
        depth: (B, 1, H, W) depth map
        pose: (B, 4, 4) relative pose
        K: (B, 3, 3) intrinsics
        K_inv: (B, 3, 3) inverse intrinsics
        
    Returns:
        pixel_coords: (B, 2, H, W) projected pixel coordinates
        valid_mask: (B, 1, H, W) validity mask
    """
    B, _, H, W = depth.shape
    device = depth.device
    
    # Backproject
    cam_points = pixel2cam(depth, K_inv)
    cam_flat = cam_points.view(B, 3, -1)
    
    # Transform
    ones = torch.ones(B, 1, H * W, device=device)
    cam_homog = torch.cat([cam_flat, ones], dim=1)
    source_cam = torch.bmm(pose[:, :3, :], cam_homog)
    source_cam = source_cam.view(B, 3, H, W)
    
    # Project
    source_depth = source_cam[:, 2:3, :, :]
    pixel_coords = cam2pixel(source_cam, K)
    
    # Validity
    x_valid = (pixel_coords[:, 0, :, :] >= 0) & (pixel_coords[:, 0, :, :] < W)
    y_valid = (pixel_coords[:, 1, :, :] >= 0) & (pixel_coords[:, 1, :, :] < H)
    depth_valid = source_depth[:, 0, :, :] > 0
    valid_mask = (x_valid & y_valid & depth_valid).unsqueeze(1).float()
    
    return pixel_coords, valid_mask
