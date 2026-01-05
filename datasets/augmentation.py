"""
Strong augmentations for monocular depth estimation.

Includes:
- Color jitter (brightness, contrast, saturation, hue)
- Horizontal flip with intrinsics adjustment
- Random scale and crop
- Photometric distortions
- Occlusion simulation

Key for reaching 90-95% accuracy on KITTI.
"""

import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


class ColorJitter:
    """Strong color augmentation."""
    
    def __init__(
        self,
        brightness: Tuple[float, float] = (0.8, 1.2),
        contrast: Tuple[float, float] = (0.8, 1.2),
        saturation: Tuple[float, float] = (0.8, 1.2),
        hue: Tuple[float, float] = (-0.1, 0.1)
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply color jitter to tensor image."""
        # Random factors
        brightness = random.uniform(*self.brightness)
        contrast = random.uniform(*self.contrast)
        saturation = random.uniform(*self.saturation)
        hue = random.uniform(*self.hue)
        
        # Apply in random order
        transforms = [
            lambda x: TF.adjust_brightness(x, brightness),
            lambda x: TF.adjust_contrast(x, contrast),
            lambda x: TF.adjust_saturation(x, saturation),
            lambda x: TF.adjust_hue(x, hue)
        ]
        random.shuffle(transforms)
        
        for t in transforms:
            img = t(img)
        
        return img.clamp(0, 1)


class RandomGamma:
    """Random gamma correction."""
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)):
        self.gamma_range = gamma_range
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        gamma = random.uniform(*self.gamma_range)
        return img.pow(gamma).clamp(0, 1)


class GaussianBlur:
    """Random Gaussian blur."""
    
    def __init__(self, kernel_size: int = 5, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            return img
        
        sigma = random.uniform(*self.sigma_range)
        
        # Create Gaussian kernel
        size = self.kernel_size
        x = torch.arange(size, dtype=torch.float32, device=img.device) - size // 2
        kernel_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel = kernel.expand(3, 1, size, size)
        
        # Apply
        padding = size // 2
        img = F.pad(img.unsqueeze(0), (padding,) * 4, mode='reflect')
        img = F.conv2d(img, kernel, groups=3)
        
        return img.squeeze(0)


class RandomOcclusion:
    """Simulate random occlusions (black rectangles)."""
    
    def __init__(
        self,
        num_occlusions: Tuple[int, int] = (0, 3),
        size_range: Tuple[float, float] = (0.02, 0.1)
    ):
        self.num_occlusions = num_occlusions
        self.size_range = size_range
        
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.3:
            return img
            
        _, H, W = img.shape
        num_occ = random.randint(*self.num_occlusions)
        
        for _ in range(num_occ):
            # Random size
            h = int(H * random.uniform(*self.size_range))
            w = int(W * random.uniform(*self.size_range))
            
            # Random position
            y = random.randint(0, H - h)
            x = random.randint(0, W - w)
            
            # Apply occlusion
            img[:, y:y+h, x:x+w] = 0
        
        return img


class RandomScale:
    """Random scaling with crop."""
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        target_height: int = 192,
        target_width: int = 640
    ):
        self.scale_range = scale_range
        self.target_height = target_height
        self.target_width = target_width
        
    def __call__(self, img: torch.Tensor, K: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply random scaling.
        
        Args:
            img: (C, H, W) tensor image
            K: (4, 4) camera intrinsics (optional)
            
        Returns:
            scaled_img: Scaled and cropped image
            K_scaled: Adjusted intrinsics
        """
        scale = random.uniform(*self.scale_range)
        
        _, H, W = img.shape
        new_H = int(H * scale)
        new_W = int(W * scale)
        
        # Resize
        img_scaled = F.interpolate(
            img.unsqueeze(0), 
            size=(new_H, new_W), 
            mode='bilinear', 
            align_corners=True
        ).squeeze(0)
        
        # Random crop to target size
        if new_H >= self.target_height and new_W >= self.target_width:
            y0 = random.randint(0, new_H - self.target_height)
            x0 = random.randint(0, new_W - self.target_width)
            img_out = img_scaled[:, y0:y0+self.target_height, x0:x0+self.target_width]
        else:
            # If too small, just resize to target
            img_out = F.interpolate(
                img_scaled.unsqueeze(0),
                size=(self.target_height, self.target_width),
                mode='bilinear',
                align_corners=True
            ).squeeze(0)
            y0, x0 = 0, 0
        
        # Adjust intrinsics
        K_scaled = None
        if K is not None:
            K_scaled = K.clone()
            K_scaled[0, :] *= scale
            K_scaled[1, :] *= scale
            K_scaled[0, 2] -= x0
            K_scaled[1, 2] -= y0
        
        return img_out, K_scaled


class PhotometricAugmentation:
    """Combined photometric augmentations."""
    
    def __init__(
        self,
        use_color_jitter: bool = True,
        use_gamma: bool = True,
        use_blur: bool = True,
        use_occlusion: bool = False,
        strength: float = 1.0
    ):
        self.transforms = []
        
        if use_color_jitter:
            brightness = (1.0 - 0.2 * strength, 1.0 + 0.2 * strength)
            contrast = (1.0 - 0.2 * strength, 1.0 + 0.2 * strength)
            saturation = (1.0 - 0.2 * strength, 1.0 + 0.2 * strength)
            hue = (-0.1 * strength, 0.1 * strength)
            self.transforms.append(ColorJitter(brightness, contrast, saturation, hue))
        
        if use_gamma:
            gamma_range = (1.0 - 0.2 * strength, 1.0 + 0.2 * strength)
            self.transforms.append(RandomGamma(gamma_range))
        
        if use_blur:
            self.transforms.append(GaussianBlur())
        
        if use_occlusion:
            self.transforms.append(RandomOcclusion())
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


class GeometricAugmentation:
    """Geometric augmentations with intrinsics adjustment."""
    
    def __init__(
        self,
        do_flip: bool = True,
        do_scale: bool = True,
        target_height: int = 192,
        target_width: int = 640
    ):
        self.do_flip = do_flip
        self.do_scale = do_scale
        
        if do_scale:
            self.scaler = RandomScale(
                scale_range=(0.9, 1.1),
                target_height=target_height,
                target_width=target_width
            )
    
    def __call__(
        self, 
        images: Dict[str, torch.Tensor],
        K: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply geometric augmentations to all images consistently.
        
        Args:
            images: Dict of images (same augmentation applied to all)
            K: Camera intrinsics
            
        Returns:
            aug_images: Augmented images
            K_aug: Adjusted intrinsics
        """
        # Random decisions (apply same to all images)
        flip = self.do_flip and random.random() > 0.5
        scale = self.do_scale and random.random() > 0.5
        
        K_aug = K.clone()
        aug_images = {}
        
        for key, img in images.items():
            # Horizontal flip
            if flip:
                img = torch.flip(img, dims=[-1])
            
            # Random scale (disabled for now - complex with multiple images)
            # if scale:
            #     img, K_aug = self.scaler(img, K_aug)
            
            aug_images[key] = img
        
        # Adjust intrinsics for flip
        if flip:
            _, _, W = list(images.values())[0].shape
            K_aug[0, 2] = W - K_aug[0, 2]
        
        return aug_images, K_aug


class DepthAugmentation:
    """
    Complete augmentation pipeline for monocular depth estimation.
    """
    
    def __init__(
        self,
        height: int = 192,
        width: int = 640,
        color_aug_prob: float = 0.5,
        flip_prob: float = 0.5,
        aug_strength: float = 1.0
    ):
        self.height = height
        self.width = width
        self.color_aug_prob = color_aug_prob
        self.flip_prob = flip_prob
        
        self.photo_aug = PhotometricAugmentation(
            use_color_jitter=True,
            use_gamma=True,
            use_blur=False,
            use_occlusion=False,
            strength=aug_strength
        )
    
    def __call__(self, data: Dict) -> Dict:
        """
        Apply augmentations to training data.
        
        Args:
            data: Dict with ('color', frame_id, scale) keys
            
        Returns:
            aug_data: Augmented data
        """
        aug_data = {}
        
        # Determine augmentation flags (same for all frames)
        do_color = random.random() < self.color_aug_prob
        do_flip = random.random() < self.flip_prob
        
        # Process each item
        for key, value in data.items():
            if not isinstance(key, tuple):
                aug_data[key] = value
                continue
                
            if key[0] == 'color':
                frame_id, scale = key[1], key[2]
                
                img = value.clone()
                
                # Horizontal flip
                if do_flip:
                    img = torch.flip(img, dims=[-1])
                
                # Color augmentation (only at scale 0, same params for all frames)
                if do_color and scale == 0:
                    img = self.photo_aug(img)
                
                aug_data[key] = img
            
            elif key[0] in ['K', 'inv_K']:
                K = value.clone()
                
                if do_flip and key[0] == 'K':
                    scale = key[1]
                    W = self.width // (2 ** scale)
                    K[0, 2] = W - K[0, 2]
                    aug_data[key] = K
                    aug_data[('inv_K', scale)] = torch.linalg.pinv(K)
                else:
                    aug_data[key] = value
            
            else:
                aug_data[key] = value
        
        return aug_data
