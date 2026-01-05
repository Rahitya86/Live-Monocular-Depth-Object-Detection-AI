"""
Dataset for loading triplets of consecutive video frames.

TODO: Swap dataset implementation here
- Currently supports a simple folder structure
- Can be extended for KITTI, Cityscapes, or custom datasets
- Modify __getitem__ to load from different formats
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MonoDataset(Dataset):
    """
    Dataset for monocular depth estimation training.
    Loads triplets of consecutive frames (t-1, t, t+1).
    
    Expected folder structure:
        data_root/
            sequence_01/
                frame_0000.png
                frame_0001.png
                ...
            sequence_02/
                ...
    """
    
    def __init__(
        self,
        data_root: str,
        height: int = 192,
        width: int = 640,
        frame_ids: List[int] = [0, -1, 1],
        num_scales: int = 4,
        is_train: bool = True,
        img_ext: str = '.png'
    ):
        """
        Args:
            data_root: Root directory containing sequences
            height: Output image height
            width: Output image width
            frame_ids: Frame offsets to load (0 = target, -1 = prev, 1 = next)
            num_scales: Number of scales for multi-scale training
            is_train: Whether this is training set (enables augmentation)
            img_ext: Image file extension
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.height = height
        self.width = width
        self.frame_ids = frame_ids
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext
        
        # Collect all valid frame triplets
        self.samples = self._collect_samples()
        
        # Image transforms
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
        
        # Augmentation for training
        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        
    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Collect all valid (sequence_path, frame_idx) pairs.
        Ensures we can load all required frame offsets.
        """
        samples = []
        
        if not self.data_root.exists():
            return samples
            
        # Find all sequences
        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue
                
            # Get sorted list of frames
            frames = sorted(seq_dir.glob(f'*{self.img_ext}'))
            num_frames = len(frames)
            
            # Determine valid indices
            min_offset = min(self.frame_ids)
            max_offset = max(self.frame_ids)
            
            for idx in range(abs(min_offset), num_frames - max_offset):
                samples.append((seq_dir, idx, [f.stem for f in frames]))
                
        return samples
    
    def __len__(self) -> int:
        return max(len(self.samples), 1)  # Return at least 1 for smoke test
    
    def __getitem__(self, index: int) -> dict:
        """
        Load a frame triplet.
        
        Returns:
            data: Dict containing:
                - ('color', frame_id, scale): Images at each scale
                - 'K': Camera intrinsics
                - 'inv_K': Inverse intrinsics
        """
        data = {}
        
        if len(self.samples) == 0:
            # Return dummy data for smoke test
            return self._get_dummy_data()
        
        seq_dir, center_idx, frame_names = self.samples[index % len(self.samples)]
        
        # Color augmentation params (consistent across triplet)
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        
        # Load frames
        for frame_id in self.frame_ids:
            frame_idx = center_idx + frame_id
            frame_name = frame_names[frame_idx]
            img_path = seq_dir / f'{frame_name}{self.img_ext}'
            
            img = self._load_image(img_path)
            
            # Apply augmentation
            if do_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Generate multi-scale images
            for scale in range(self.num_scales):
                K = self._get_intrinsics(scale)
                
                if scale == 0:
                    scaled_img = img.resize((self.width, self.height), Image.LANCZOS)
                else:
                    s = 2 ** scale
                    scaled_img = img.resize(
                        (self.width // s, self.height // s), Image.LANCZOS
                    )
                
                # Convert to tensor
                tensor = self.to_tensor(scaled_img)
                
                # Color augmentation
                if do_color_aug and scale == 0:
                    tensor = self._color_augment(tensor)
                
                data[('color', frame_id, scale)] = tensor
        
        # Add intrinsics
        for scale in range(self.num_scales):
            K = self._get_intrinsics(scale)
            data[('K', scale)] = K
            data[('inv_K', scale)] = torch.inverse(K)
        
        return data
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        img = Image.open(path).convert('RGB')
        return img
    
    def _get_intrinsics(self, scale: int = 0) -> torch.Tensor:
        """
        Get camera intrinsics matrix.
        Default values assume KITTI-like setup.
        
        TODO: Load actual intrinsics from calibration file
        """
        # Scale factors for current resolution
        s = 2 ** scale
        h = self.height // s
        w = self.width // s
        
        # Default focal length (0.58 * width)
        fx = 0.58 * w
        fy = 0.58 * h * (self.width / self.height)
        
        # Principal point at center
        cx = 0.5 * w
        cy = 0.5 * h
        
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return K
    
    def _color_augment(self, img: torch.Tensor) -> torch.Tensor:
        """Apply color augmentation."""
        transforms = T.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        return transforms(img)
    
    def _get_dummy_data(self) -> dict:
        """Generate dummy data for smoke test."""
        data = {}
        
        for frame_id in self.frame_ids:
            for scale in range(self.num_scales):
                s = 2 ** scale
                h = self.height // s
                w = self.width // s
                data[('color', frame_id, scale)] = torch.rand(3, h, w)
        
        for scale in range(self.num_scales):
            K = self._get_intrinsics(scale)
            data[('K', scale)] = K
            data[('inv_K', scale)] = torch.inverse(K)
        
        return data


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing/smoke tests.
    Generates random image triplets with known camera motion.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        height: int = 192,
        width: int = 640,
        num_scales: int = 4
    ):
        super().__init__()
        self.num_samples = num_samples
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.frame_ids = [0, -1, 1]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        data = {}
        
        # Generate base image
        base_img = torch.rand(3, self.height, self.width)
        
        # Generate slightly shifted versions for temporal frames
        for frame_id in self.frame_ids:
            for scale in range(self.num_scales):
                s = 2 ** scale
                h = self.height // s
                w = self.width // s
                
                if scale == 0:
                    img = base_img + 0.1 * frame_id * torch.randn_like(base_img)
                else:
                    img = torch.nn.functional.interpolate(
                        base_img.unsqueeze(0),
                        size=(h, w),
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(0)
                    
                data[('color', frame_id, scale)] = img.clamp(0, 1)
        
        # Intrinsics
        for scale in range(self.num_scales):
            s = 2 ** scale
            h = self.height // s
            w = self.width // s
            
            fx = 0.58 * w
            fy = 0.58 * h
            cx = 0.5 * w
            cy = 0.5 * h
            
            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32)
            
            data[('K', scale)] = K
            data[('inv_K', scale)] = torch.inverse(K)
        
        return data
