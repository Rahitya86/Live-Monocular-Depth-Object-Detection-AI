"""
SSIM (Structural Similarity Index) loss with optimizations.
Uses Gaussian weighting for better perceptual similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel."""
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM."""
    _1D_window = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIM(nn.Module):
    """
    Structural Similarity Index with Gaussian weighting.
    
    Uses proper Gaussian kernel instead of simple average pooling
    for better perceptual quality measurement.
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)
        
        # Constants for numerical stability
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images.
        
        Args:
            img1: (B, C, H, W) first image
            img2: (B, C, H, W) second image
            
        Returns:
            ssim_map: (B, 1, H, W) SSIM map
        """
        channel = img1.size(1)
        
        # Create window for current channel count
        if channel != self.channel or self.window.device != img1.device:
            self.window = create_window(self.window_size, channel).to(img1.device)
            self.channel = channel
        
        pad = self.window_size // 2
        
        # Compute means
        mu1 = F.conv2d(img1, self.window, padding=pad, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=channel) - mu1_mu2
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        ssim_map = numerator / denominator
        
        # Average across channels
        return ssim_map.mean(dim=1, keepdim=True)


class SSIMLoss(nn.Module):
    """SSIM loss module (1 - SSIM)."""
    
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.ssim = SSIM(window_size)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (lower = more similar)."""
        return (1 - self.ssim(pred, target)).clamp(0, 2) / 2


def compute_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM loss with proper Gaussian weighting.
    
    Returns:
        loss: (B, 1, H, W) per-pixel SSIM loss
    """
    ssim = SSIM(window_size=11)
    return (1 - ssim(pred, target)).clamp(0, 2) / 2
