"""
Transformation utilities for camera pose.
Handles 6-DoF pose representation and transformation matrices.
"""

import torch
import torch.nn.functional as F


def euler_to_rotation_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix.
    Uses ZYX convention (yaw-pitch-roll).
    
    Args:
        euler_angles: (B, 3) Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        R: (B, 3, 3) rotation matrices
    """
    B = euler_angles.shape[0]
    device = euler_angles.device
    
    roll = euler_angles[:, 0]   # rotation around X
    pitch = euler_angles[:, 1]  # rotation around Y
    yaw = euler_angles[:, 2]    # rotation around Z
    
    # Precompute sin/cos
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    
    # Build rotation matrix (ZYX convention)
    zeros = torch.zeros(B, device=device)
    ones = torch.ones(B, device=device)
    
    # Rz (yaw)
    Rz = torch.stack([
        torch.stack([cy, -sy, zeros], dim=1),
        torch.stack([sy, cy, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # (B, 3, 3)
    
    # Ry (pitch)
    Ry = torch.stack([
        torch.stack([cp, zeros, sp], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sp, zeros, cp], dim=1)
    ], dim=1)  # (B, 3, 3)
    
    # Rx (roll)
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cr, -sr], dim=1),
        torch.stack([zeros, sr, cr], dim=1)
    ], dim=1)  # (B, 3, 3)
    
    # R = Rz @ Ry @ Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    return R


def axisangle_to_rotation_matrix(axisangle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula.
    
    Args:
        axisangle: (B, 3) axis-angle rotation vector
        
    Returns:
        R: (B, 3, 3) rotation matrices
    """
    B = axisangle.shape[0]
    device = axisangle.device
    
    # Get angle (norm of axis-angle vector)
    theta = torch.norm(axisangle, dim=1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    
    # Normalize to get axis (handle zero rotation)
    axis = axisangle / (theta.squeeze(-1) + 1e-8)  # (B, 3)
    
    # Skew-symmetric matrix
    zeros = torch.zeros(B, device=device)
    K = torch.stack([
        torch.stack([zeros, -axis[:, 2], axis[:, 1]], dim=1),
        torch.stack([axis[:, 2], zeros, -axis[:, 0]], dim=1),
        torch.stack([-axis[:, 1], axis[:, 0], zeros], dim=1)
    ], dim=1)  # (B, 3, 3)
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    R = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    return R


def pose_vec_to_matrix(translation: torch.Tensor, rotation: torch.Tensor, 
                       rotation_mode: str = 'axisangle') -> torch.Tensor:
    """
    Convert pose vector (translation + rotation) to 4x4 transformation matrix.
    
    Args:
        translation: (B, 3) translation vector
        rotation: (B, 3) rotation (axis-angle or euler)
        rotation_mode: 'axisangle' or 'euler'
        
    Returns:
        T: (B, 4, 4) transformation matrix
    """
    B = translation.shape[0]
    device = translation.device
    
    # Get rotation matrix
    if rotation_mode == 'axisangle':
        R = axisangle_to_rotation_matrix(rotation)
    elif rotation_mode == 'euler':
        R = euler_to_rotation_matrix(rotation)
    else:
        raise ValueError(f"Unknown rotation mode: {rotation_mode}")
    
    # Build 4x4 transformation matrix
    T = torch.zeros(B, 4, 4, device=device)
    T[:, :3, :3] = R
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.0
    
    return T


def invert_pose(T: torch.Tensor) -> torch.Tensor:
    """
    Invert a 4x4 transformation matrix.
    
    Args:
        T: (B, 4, 4) transformation matrix
        
    Returns:
        T_inv: (B, 4, 4) inverse transformation matrix
    """
    B = T.shape[0]
    device = T.device
    
    R = T[:, :3, :3]  # (B, 3, 3)
    t = T[:, :3, 3:]  # (B, 3, 1)
    
    R_inv = R.transpose(1, 2)  # R^T
    t_inv = -torch.bmm(R_inv, t)  # -R^T @ t
    
    T_inv = torch.zeros(B, 4, 4, device=device)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    T_inv[:, 3, 3] = 1.0
    
    return T_inv
