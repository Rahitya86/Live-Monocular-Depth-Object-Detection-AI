"""
Edge-aware smoothness loss for depth regularization.

Optimizations:
- Proper normalization by mean disparity
- Edge-aware weighting from image gradients
- Multi-scale computation with proper weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


def compute_smoothness_loss(
    disp: torch.Tensor,
    img: torch.Tensor,
    edge_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute edge-aware smoothness loss.
    
    Encourages smooth depth while preserving edges aligned with image.
    
    Args:
        disp: (B, 1, H, W) disparity map
        img: (B, 3, H, W) corresponding RGB image
        edge_weight: Weight for edge-aware term (higher = sharper edges)
        
    Returns:
        loss: Scalar smoothness loss
    """
    # Normalize disparity by mean to handle scale ambiguity
    mean_disp = disp.mean(dim=[2, 3], keepdim=True)
    disp_normalized = disp / (mean_disp + 1e-7)
    
    # Compute disparity gradients
    grad_disp_x = torch.abs(disp_normalized[:, :, :, :-1] - disp_normalized[:, :, :, 1:])
    grad_disp_y = torch.abs(disp_normalized[:, :, :-1, :] - disp_normalized[:, :, 1:, :])
    
    # Compute image gradients (mean across RGB channels)
    grad_img_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
    grad_img_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
    
    # Edge-aware weights: less smoothing at image edges
    weight_x = torch.exp(-edge_weight * grad_img_x)
    weight_y = torch.exp(-edge_weight * grad_img_y)
    
    # Weighted smoothness
    smoothness_x = grad_disp_x * weight_x
    smoothness_y = grad_disp_y * weight_y
    
    return smoothness_x.mean() + smoothness_y.mean()


def compute_second_order_smoothness(
    disp: torch.Tensor,
    img: torch.Tensor,
    edge_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute second-order smoothness for piece-wise planar surfaces.
    
    Second-order smoothness allows linear depth changes (planes)
    while penalizing curvature.
    """
    # Normalize
    mean_disp = disp.mean(dim=[2, 3], keepdim=True)
    disp_norm = disp / (mean_disp + 1e-7)
    
    # Second-order gradients (Laplacian approximation)
    grad_x = disp_norm[:, :, :, 1:] - disp_norm[:, :, :, :-1]
    grad_xx = torch.abs(grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1])
    
    grad_y = disp_norm[:, :, 1:, :] - disp_norm[:, :, :-1, :]
    grad_yy = torch.abs(grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :])
    
    # Image gradients for edge-awareness
    grad_img_x = torch.abs(img[:, :, :, 1:-1] - img[:, :, :, :-2]).mean(dim=1, keepdim=True)
    grad_img_y = torch.abs(img[:, :, 1:-1, :] - img[:, :, :-2, :]).mean(dim=1, keepdim=True)
    
    weight_x = torch.exp(-edge_weight * grad_img_x)
    weight_y = torch.exp(-edge_weight * grad_img_y)
    
    # Resize weights to match gradient sizes
    weight_x = weight_x[:, :, :, :grad_xx.shape[3]]
    weight_y = weight_y[:, :, :grad_yy.shape[2], :]
    
    smooth_x = (grad_xx * weight_x).mean()
    smooth_y = (grad_yy * weight_y).mean()
    
    return smooth_x + smooth_y


def compute_multiscale_smoothness(
    disps: Dict,
    imgs: Dict,
    scales: List[int] = [0, 1, 2, 3],
    edge_weight: float = 1.0
) -> torch.Tensor:
    """
    Compute smoothness loss across multiple scales.
    
    Args:
        disps: Dict of disparity maps {('disp', scale): tensor}
        imgs: Dict of images at each scale {scale: tensor}
        scales: List of scales
        edge_weight: Edge-aware weight
        
    Returns:
        loss: Total smoothness loss
    """
    total_loss = 0.0
    
    for scale in scales:
        if ('disp', scale) not in disps or scale not in imgs:
            continue
            
        disp = disps[('disp', scale)]
        img = imgs[scale]
        
        # Ensure same spatial size
        if disp.shape[2:] != img.shape[2:]:
            img = F.interpolate(img, size=disp.shape[2:], mode='bilinear', align_corners=True)
        
        # Weight lower resolution scales less
        scale_weight = 1.0 / (2 ** scale)
        
        loss = compute_smoothness_loss(disp, img, edge_weight)
        total_loss = total_loss + scale_weight * loss
    
    return total_loss


def get_scaled_images(img: torch.Tensor, scales: List[int] = [0, 1, 2, 3]) -> Dict:
    """
    Generate multi-scale versions of an image.
    
    Args:
        img: (B, 3, H, W) input image
        scales: List of scales
        
    Returns:
        scaled_imgs: Dict of scaled images {scale: tensor}
    """
    scaled_imgs = {}
    
    for scale in scales:
        if scale == 0:
            scaled_imgs[scale] = img
        else:
            factor = 2 ** scale
            h, w = img.shape[2] // factor, img.shape[3] // factor
            scaled_imgs[scale] = F.interpolate(
                img, size=(h, w), mode='bilinear', align_corners=True
            )
    
    return scaled_imgs


class SmoothnessLoss(nn.Module):
    """Edge-aware smoothness loss module."""
    
    def __init__(self, edge_weight: float = 1.0, second_order: bool = False):
        super().__init__()
        self.edge_weight = edge_weight
        self.second_order = second_order
        
    def forward(self, disp: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        if self.second_order:
            return compute_second_order_smoothness(disp, img, self.edge_weight)
        return compute_smoothness_loss(disp, img, self.edge_weight)
