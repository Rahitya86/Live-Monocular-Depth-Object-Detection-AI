"""
Utility functions for training and visualization.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: str,
    epoch: int,
    depth_net: nn.Module,
    pose_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    best_loss: float = float('inf')
):
    """
    Save training checkpoint.
    
    Args:
        path: Path to save checkpoint
        epoch: Current epoch
        depth_net: Depth estimation network
        pose_net: Pose estimation network
        optimizer: Optimizer state
        scheduler: Optional LR scheduler
        best_loss: Best validation loss so far
    """
    checkpoint = {
        'epoch': epoch,
        'depth_net_state_dict': depth_net.state_dict(),
        'pose_net_state_dict': pose_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    depth_net: nn.Module,
    pose_net: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Load training checkpoint.
    
    Args:
        path: Path to checkpoint file
        depth_net: Depth estimation network
        pose_net: Pose estimation network
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        device: Device to load to
        
    Returns:
        checkpoint: Dict with epoch and best_loss
    """
    checkpoint = torch.load(path, map_location=device)
    
    depth_net.load_state_dict(checkpoint['depth_net_state_dict'])
    pose_net.load_state_dict(checkpoint['pose_net_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {path} (epoch {checkpoint['epoch']})")
    
    return {
        'epoch': checkpoint['epoch'],
        'best_loss': checkpoint.get('best_loss', float('inf'))
    }


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize_depth(depth: torch.Tensor, min_depth: float = 0.1, max_depth: float = 100.0) -> np.ndarray:
    """
    Convert depth map to colorized visualization.
    
    Args:
        depth: (H, W) or (1, H, W) depth tensor
        min_depth: Minimum depth for normalization
        max_depth: Maximum depth for normalization
        
    Returns:
        colored: (H, W, 3) RGB array (0-255)
    """
    import matplotlib.pyplot as plt
    
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    
    depth_np = depth.detach().cpu().numpy()
    
    # Normalize to [0, 1]
    depth_normalized = (depth_np - min_depth) / (max_depth - min_depth)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # Apply colormap (magma for depth)
    cmap = plt.cm.magma
    colored = cmap(depth_normalized)[:, :, :3]  # Remove alpha
    colored = (colored * 255).astype(np.uint8)
    
    return colored


def colorize_disparity(disp: torch.Tensor) -> np.ndarray:
    """
    Convert disparity map to colorized visualization.
    
    Args:
        disp: (H, W) or (1, H, W) disparity tensor
        
    Returns:
        colored: (H, W, 3) RGB array (0-255)
    """
    import matplotlib.pyplot as plt
    
    if disp.dim() == 3:
        disp = disp.squeeze(0)
    
    disp_np = disp.detach().cpu().numpy()
    
    # Normalize to [0, 1]
    disp_normalized = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.cm.plasma
    colored = cmap(disp_normalized)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    
    return colored


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image for visualization.
    
    Args:
        tensor: (C, H, W) or (B, C, H, W) tensor in [0, 1]
        
    Returns:
        image: (H, W, 3) numpy array in [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img
