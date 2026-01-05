#!/usr/bin/env python3
"""
Visualization script for monocular depth estimation.
Displays depth predictions on images.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from models import DepthNet
from utils import colorize_depth, colorize_disparity, tensor_to_image, load_checkpoint


def load_image(path: str, height: int = 192, width: int = 640) -> tuple:
    """
    Load and preprocess an image.
    
    Args:
        path: Path to image file
        height: Target height
        width: Target width
        
    Returns:
        tensor: (1, 3, H, W) normalized tensor
        original: Original PIL image
    """
    img = Image.open(path).convert('RGB')
    original = img.copy()
    
    # Resize
    img = img.resize((width, height), Image.LANCZOS)
    
    # Convert to tensor and normalize
    to_tensor = T.ToTensor()
    tensor = to_tensor(img).unsqueeze(0)  # (1, 3, H, W)
    
    return tensor, original


@torch.no_grad()
def predict_depth(
    model: DepthNet,
    image: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Predict depth from a single image.
    
    Args:
        model: DepthNet model
        image: (1, 3, H, W) input tensor
        device: Computation device
        
    Returns:
        outputs: Dict with depth and disparity maps
    """
    model.eval()
    image = image.to(device)
    outputs = model(image)
    return outputs


def visualize_prediction(
    image: torch.Tensor,
    outputs: dict,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize depth prediction.
    
    Args:
        image: (1, 3, H, W) input image tensor
        outputs: Model outputs with depth and disparity
        save_path: Optional path to save visualization
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    img_np = tensor_to_image(image)
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Disparity (scale 0)
    disp = outputs[('disp', 0)][0]  # Remove batch dim
    disp_colored = colorize_disparity(disp)
    axes[0, 1].imshow(disp_colored)
    axes[0, 1].set_title('Disparity')
    axes[0, 1].axis('off')
    
    # Depth (scale 0)
    depth = outputs[('depth', 0)][0]
    depth_colored = colorize_depth(depth, min_depth=0.1, max_depth=100.0)
    axes[1, 0].imshow(depth_colored)
    axes[1, 0].set_title('Depth')
    axes[1, 0].axis('off')
    
    # Depth histogram
    depth_np = depth.squeeze().cpu().numpy()
    axes[1, 1].hist(depth_np.flatten(), bins=100, color='blue', alpha=0.7)
    axes[1, 1].set_xlabel('Depth (m)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Depth Distribution')
    axes[1, 1].set_xlim(0, 50)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def visualize_multiscale(outputs: dict, save_path: str = None, show: bool = True):
    """
    Visualize multi-scale depth predictions.
    
    Args:
        outputs: Model outputs with multi-scale depth/disparity
        save_path: Optional path to save visualization
        show: Whether to display
    """
    scales = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, len(scales), figsize=(16, 8))
    
    for i, scale in enumerate(scales):
        # Disparity
        if ('disp', scale) in outputs:
            disp = outputs[('disp', scale)][0]
            disp_colored = colorize_disparity(disp)
            axes[0, i].imshow(disp_colored)
            axes[0, i].set_title(f'Disparity (1/{2**scale})')
            axes[0, i].axis('off')
        
        # Depth
        if ('depth', scale) in outputs:
            depth = outputs[('depth', scale)][0]
            depth_colored = colorize_depth(depth)
            axes[1, i].imshow(depth_colored)
            axes[1, i].set_title(f'Depth (1/{2**scale})')
            axes[1, i].axis('off')
    
    plt.suptitle('Multi-scale Predictions')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


def save_depth_map(depth: torch.Tensor, save_path: str, format: str = 'png'):
    """
    Save depth map to file.
    
    Args:
        depth: (1, H, W) or (H, W) depth tensor
        save_path: Output path
        format: 'png' for visualization, 'npy' for raw values
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    
    depth_np = depth.cpu().numpy()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'npy':
        np.save(save_path, depth_np)
    else:
        depth_colored = colorize_depth(depth)
        Image.fromarray(depth_colored).save(save_path)
    
    print(f"Saved depth map to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize depth predictions')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--height', type=int, default=192,
                        help='Input height')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    depth_net = DepthNet(
        encoder_name='resnet18',
        pretrained=False
    ).to(device)
    
    # Load checkpoint if exists
    if Path(args.checkpoint).exists():
        # Create dummy pose_net for loading
        from models import PoseNet
        pose_net = PoseNet(num_input_images=2).to(device)
        load_checkpoint(args.checkpoint, depth_net, pose_net, device=device)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint found, using random weights")
    
    # Load image
    image_tensor, original_image = load_image(args.image, args.height, args.width)
    print(f"Loaded image: {args.image}")
    
    # Predict depth
    outputs = predict_depth(depth_net, image_tensor, device)
    
    # Visualize
    image_name = Path(args.image).stem
    output_dir = Path(args.output)
    
    visualize_prediction(
        image_tensor,
        outputs,
        save_path=str(output_dir / f'{image_name}_prediction.png'),
        show=not args.no_show
    )
    
    visualize_multiscale(
        outputs,
        save_path=str(output_dir / f'{image_name}_multiscale.png'),
        show=not args.no_show
    )
    
    # Save depth map
    save_depth_map(
        outputs[('depth', 0)][0],
        str(output_dir / f'{image_name}_depth.png')
    )
    save_depth_map(
        outputs[('depth', 0)][0],
        str(output_dir / f'{image_name}_depth.npy'),
        format='npy'
    )
    
    print("Visualization complete!")


if __name__ == '__main__':
    main()
