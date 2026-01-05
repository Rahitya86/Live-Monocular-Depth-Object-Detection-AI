"""
Stereo supervision loss for monocular depth estimation.

Adds stereo consistency constraint using left-right image pairs.
This significantly improves accuracy when stereo data is available.

Key for reaching 90-95% accuracy on KITTI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class StereoConsistencyLoss(nn.Module):
    """
    Stereo consistency loss using left-right image pairs.
    
    Warps left image to right view (and vice versa) using
    predicted depth and known stereo baseline.
    
    Loss = |I_L - warp(I_R, D_L)| + |I_R - warp(I_L, D_R)|
    """
    
    def __init__(
        self,
        ssim_weight: float = 0.85,
        use_auto_mask: bool = True,
        scales: List[int] = [0, 1, 2, 3],
        baseline: float = 0.54  # KITTI stereo baseline in meters
    ):
        """
        Args:
            ssim_weight: Weight for SSIM in photometric loss
            use_auto_mask: Use auto-masking for static regions
            scales: Scales to compute loss at
            baseline: Stereo baseline in meters
        """
        super().__init__()
        self.ssim_weight = ssim_weight
        self.use_auto_mask = use_auto_mask
        self.scales = scales
        self.baseline = baseline
        
    def forward(
        self,
        left_depth: torch.Tensor,
        right_depth: torch.Tensor,
        left_img: torch.Tensor,
        right_img: torch.Tensor,
        K: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute stereo consistency loss.
        
        Args:
            left_depth: (B, 1, H, W) left image depth
            right_depth: (B, 1, H, W) right image depth
            left_img: (B, 3, H, W) left image
            right_img: (B, 3, H, W) right image
            K: (B, 4, 4) camera intrinsics
            
        Returns:
            losses: Dict with individual losses
        """
        losses = {}
        
        B, _, H, W = left_img.shape
        device = left_img.device
        
        # Get focal length from intrinsics
        fx = K[:, 0, 0]  # (B,)
        
        # Convert depth to disparity
        # disparity = baseline * fx / depth
        left_disp = self.baseline * fx.view(B, 1, 1, 1) / left_depth.clamp(min=0.1)
        right_disp = self.baseline * fx.view(B, 1, 1, 1) / right_depth.clamp(min=0.1)
        
        # Normalize disparity to pixel coordinates
        left_disp_px = left_disp / W  # Now in [0, 1] range roughly
        right_disp_px = right_disp / W
        
        # Create sampling grids
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Warp right to left view
        # For left image, shift right by disparity
        grid_l2r = grid.clone()
        grid_l2r[:, :, :, 0] = grid_l2r[:, :, :, 0] - 2 * left_disp_px.squeeze(1)
        
        warped_right = F.grid_sample(
            right_img, grid_l2r,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Warp left to right view
        grid_r2l = grid.clone()
        grid_r2l[:, :, :, 0] = grid_r2l[:, :, :, 0] + 2 * right_disp_px.squeeze(1)
        
        warped_left = F.grid_sample(
            left_img, grid_r2l,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        # Compute photometric losses
        loss_l2r = self._photometric_loss(left_img, warped_right)
        loss_r2l = self._photometric_loss(right_img, warped_left)
        
        # Auto-masking: compare with identity (no warping)
        if self.use_auto_mask:
            identity_loss_l = self._photometric_loss(left_img, right_img)
            identity_loss_r = self._photometric_loss(right_img, left_img)
            
            # Take minimum
            loss_l2r = torch.minimum(loss_l2r, identity_loss_l + 1e-5)
            loss_r2l = torch.minimum(loss_r2l, identity_loss_r + 1e-5)
        
        # Validity masks (inside image bounds)
        valid_l2r = (grid_l2r[:, :, :, 0].abs() <= 1).float()
        valid_r2l = (grid_r2l[:, :, :, 0].abs() <= 1).float()
        
        # Compute mean losses
        losses['stereo_l2r'] = (loss_l2r * valid_l2r.unsqueeze(1)).sum() / (valid_l2r.sum() + 1e-7)
        losses['stereo_r2l'] = (loss_r2l * valid_r2l.unsqueeze(1)).sum() / (valid_r2l.sum() + 1e-7)
        losses['stereo_total'] = losses['stereo_l2r'] + losses['stereo_r2l']
        
        # Store warped images for visualization
        losses['warped_right'] = warped_right
        losses['warped_left'] = warped_left
        
        return losses
    
    def _photometric_loss(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute photometric loss (L1 + SSIM)."""
        # L1 loss
        l1_loss = torch.abs(pred - target)
        
        # SSIM loss
        ssim_loss = 1 - self._ssim(pred, target)
        
        # Combined loss
        loss = self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * l1_loss.mean(dim=1, keepdim=True)
        
        return loss
    
    def _ssim(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window_size: int = 3
    ) -> torch.Tensor:
        """Compute SSIM."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Mean pooling
        pad = window_size // 2
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1, padding=pad) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, window_size, stride=1, padding=pad) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_xy
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim.mean(dim=1, keepdim=True)


class LRConsistencyLoss(nn.Module):
    """
    Left-Right depth consistency loss.
    
    Enforces that warped depth from left matches right depth
    and vice versa.
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        left_disp: torch.Tensor,
        right_disp: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute LR consistency loss.
        
        Args:
            left_disp: (B, 1, H, W) left disparity
            right_disp: (B, 1, H, W) right disparity
            
        Returns:
            loss: Scalar LR consistency loss
        """
        B, _, H, W = left_disp.shape
        device = left_disp.device
        
        # Create sampling grid
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        # Warp right disparity to left view
        grid_r2l = grid.clone()
        grid_r2l[:, :, :, 0] = grid_r2l[:, :, :, 0] + 2 * left_disp.squeeze(1) / W
        
        warped_right_disp = F.grid_sample(
            right_disp, grid_r2l,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Warp left disparity to right view
        grid_l2r = grid.clone()
        grid_l2r[:, :, :, 0] = grid_l2r[:, :, :, 0] - 2 * right_disp.squeeze(1) / W
        
        warped_left_disp = F.grid_sample(
            left_disp, grid_l2r,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Validity masks
        valid_r2l = (grid_r2l[:, :, :, 0].abs() <= 1).float().unsqueeze(1)
        valid_l2r = (grid_l2r[:, :, :, 0].abs() <= 1).float().unsqueeze(1)
        
        # Consistency losses
        loss_lr = torch.abs(left_disp - warped_right_disp) * valid_r2l
        loss_rl = torch.abs(right_disp - warped_left_disp) * valid_l2r
        
        loss = loss_lr.sum() / (valid_r2l.sum() + 1e-7) + \
               loss_rl.sum() / (valid_l2r.sum() + 1e-7)
        
        return self.weight * loss


class DisparitySmoothnessLoss(nn.Module):
    """
    Edge-aware disparity smoothness loss.
    
    Encourages smooth disparities except at image edges.
    """
    
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
        
    def forward(
        self,
        disp: torch.Tensor,
        img: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-aware smoothness loss.
        
        Args:
            disp: (B, 1, H, W) disparity
            img: (B, 3, H, W) image
            
        Returns:
            loss: Scalar smoothness loss
        """
        # Normalize disparity
        mean_disp = disp.mean(dim=[2, 3], keepdim=True)
        disp_norm = disp / (mean_disp + 1e-7)
        
        # Gradients
        grad_disp_x = torch.abs(disp_norm[:, :, :, :-1] - disp_norm[:, :, :, 1:])
        grad_disp_y = torch.abs(disp_norm[:, :, :-1, :] - disp_norm[:, :, 1:, :])
        
        grad_img_x = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]).mean(dim=1, keepdim=True)
        grad_img_y = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]).mean(dim=1, keepdim=True)
        
        # Edge-aware weights
        weight_x = torch.exp(-grad_img_x)
        weight_y = torch.exp(-grad_img_y)
        
        # Weighted smoothness
        loss = (grad_disp_x * weight_x).mean() + (grad_disp_y * weight_y).mean()
        
        return self.weight * loss
