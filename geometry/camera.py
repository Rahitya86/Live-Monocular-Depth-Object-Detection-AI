"""
Camera intrinsics and projection utilities.
Handles conversion between pixel coordinates and camera coordinates.
"""

import torch
import torch.nn.functional as F


def get_intrinsics(fx: float, fy: float, cx: float, cy: float, device: torch.device = None) -> torch.Tensor:
    """
    Create a camera intrinsics matrix K.
    
    Args:
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        device: Target device
        
    Returns:
        K: (3, 3) intrinsics matrix
    """
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    return K


def get_scaled_intrinsics(K: torch.Tensor, scale_x: float, scale_y: float) -> torch.Tensor:
    """
    Scale intrinsics matrix for different image resolutions.
    
    Args:
        K: (B, 3, 3) or (3, 3) intrinsics matrix
        scale_x: Scale factor in x direction
        scale_y: Scale factor in y direction
        
    Returns:
        K_scaled: Scaled intrinsics matrix
    """
    K_scaled = K.clone()
    if K.dim() == 2:
        K_scaled[0, 0] *= scale_x
        K_scaled[1, 1] *= scale_y
        K_scaled[0, 2] *= scale_x
        K_scaled[1, 2] *= scale_y
    else:  # Batched
        K_scaled[:, 0, 0] *= scale_x
        K_scaled[:, 1, 1] *= scale_y
        K_scaled[:, 0, 2] *= scale_x
        K_scaled[:, 1, 2] *= scale_y
    return K_scaled


def pixel2cam(depth: torch.Tensor, K_inv: torch.Tensor) -> torch.Tensor:
    """
    Convert pixel coordinates to camera coordinates using depth.
    Backproject 2D image points to 3D camera space.
    
    Args:
        depth: (B, 1, H, W) depth map
        K_inv: (B, 3, 3) inverse intrinsics matrix
        
    Returns:
        cam_points: (B, 3, H, W) 3D points in camera coordinates
    """
    B, _, H, W = depth.shape
    device = depth.device
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Homogeneous pixel coordinates: (3, H*W)
    ones = torch.ones_like(x)
    pixel_coords = torch.stack([x, y, ones], dim=0).view(3, -1)  # (3, H*W)
    
    # Expand for batch
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1)  # (B, 3, H*W)
    
    # Backproject: cam_coords = K_inv @ pixel_coords
    cam_coords = torch.bmm(K_inv, pixel_coords)  # (B, 3, H*W)
    
    # Scale by depth
    depth_flat = depth.view(B, 1, -1)  # (B, 1, H*W)
    cam_points = cam_coords * depth_flat  # (B, 3, H*W)
    
    return cam_points.view(B, 3, H, W)


def cam2pixel(cam_points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Project 3D camera points to 2D pixel coordinates.
    
    Args:
        cam_points: (B, 3, H, W) 3D points in camera coordinates
        K: (B, 3, 3) intrinsics matrix
        
    Returns:
        pixel_coords: (B, 2, H, W) projected pixel coordinates (x, y)
    """
    B, _, H, W = cam_points.shape
    
    # Reshape to (B, 3, H*W)
    cam_flat = cam_points.view(B, 3, -1)
    
    # Project: pixel = K @ cam
    pixel_homog = torch.bmm(K, cam_flat)  # (B, 3, H*W)
    
    # Normalize by depth (avoid division by zero)
    depth = pixel_homog[:, 2:3, :].clamp(min=1e-6)
    pixel_coords = pixel_homog[:, :2, :] / depth  # (B, 2, H*W)
    
    return pixel_coords.view(B, 2, H, W)


def normalize_coords(pixel_coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Normalize pixel coordinates to [-1, 1] for grid_sample.
    
    Args:
        pixel_coords: (B, 2, H, W) pixel coordinates (x, y)
        H: Image height
        W: Image width
        
    Returns:
        normalized_coords: (B, H, W, 2) normalized coordinates for grid_sample
    """
    B = pixel_coords.shape[0]
    
    # Normalize to [-1, 1]
    x = pixel_coords[:, 0, :, :]  # (B, H, W)
    y = pixel_coords[:, 1, :, :]  # (B, H, W)
    
    x_norm = 2.0 * x / (W - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0
    
    # Stack as (B, H, W, 2) for grid_sample
    return torch.stack([x_norm, y_norm], dim=-1)
