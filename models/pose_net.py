"""
PoseNet: 6-DoF relative pose estimation network.
Predicts camera motion between consecutive frames.

Optimizations:
- ResNet-based encoder for better feature extraction
- Proper pose scaling
- Support for multiple frame pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple


def rot_from_axisangle(axisangle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix using Rodrigues formula.
    
    Args:
        axisangle: (B, 3) axis-angle representation
        
    Returns:
        rot: (B, 3, 3) rotation matrix
    """
    angle = torch.norm(axisangle, dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
    axis = axisangle / (angle.squeeze(-1) + 1e-8)  # (B, 3)
    
    # Rodrigues formula
    cos_angle = torch.cos(angle)  # (B, 1, 1)
    sin_angle = torch.sin(angle)  # (B, 1, 1)
    
    # Skew symmetric matrix
    K = torch.zeros(axisangle.shape[0], 3, 3, device=axisangle.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    # R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=axisangle.device).unsqueeze(0)
    rot = I + sin_angle * K + (1 - cos_angle) * torch.bmm(K, K)
    
    return rot


def transformation_from_parameters(
    axisangle: torch.Tensor, 
    translation: torch.Tensor,
    invert: bool = False
) -> torch.Tensor:
    """
    Convert axis-angle + translation to 4x4 transformation matrix.
    
    Args:
        axisangle: (B, 3) rotation as axis-angle
        translation: (B, 3) translation vector
        invert: If True, return inverse transformation
        
    Returns:
        T: (B, 4, 4) transformation matrix
    """
    B = axisangle.shape[0]
    device = axisangle.device
    
    R = rot_from_axisangle(axisangle)
    t = translation.unsqueeze(-1)  # (B, 3, 1)
    
    # Build 4x4 matrix
    T = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3:4] = t
    
    if invert:
        T = torch.inverse(T)
    
    return T


class PoseEncoder(nn.Module):
    """
    ResNet-based encoder for pose estimation.
    Uses pretrained weights for better feature extraction.
    """
    
    def __init__(self, num_input_images: int = 2, pretrained: bool = True):
        super().__init__()
        
        self.num_input_images = num_input_images
        
        # Use ResNet18 as backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Modify first conv to accept concatenated images
        self.conv1 = nn.Conv2d(
            3 * num_input_images, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize from pretrained weights (average across input channels)
        if pretrained:
            pretrained_weight = resnet.conv1.weight.data
            self.conv1.weight.data = pretrained_weight.repeat(1, num_input_images, 1, 1) / num_input_images
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.num_ch_enc = 512
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PoseDecoder(nn.Module):
    """
    Decoder for 6-DoF pose prediction.
    """
    
    def __init__(self, num_ch_enc: int, num_frames: int = 1):
        super().__init__()
        
        self.num_frames = num_frames
        
        self.squeeze = nn.Conv2d(num_ch_enc, 256, 1)
        self.pose_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6 * num_frames, 1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.squeeze(features)
        x = self.pose_conv(x)
        x = x.mean(dim=[2, 3])  # Global average pooling
        
        x = x.view(-1, self.num_frames, 6)
        x = 0.01 * x  # Scale down for stability
        
        axisangle = x[..., :3]
        translation = x[..., 3:]
        
        return axisangle, translation


class PoseNet(nn.Module):
    """
    Complete pose estimation network.
    Predicts relative 6-DoF pose from consecutive frames.
    """
    
    def __init__(self, num_input_images: int = 2, pretrained: bool = True):
        super().__init__()
        
        self.encoder = PoseEncoder(num_input_images, pretrained)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_frames=num_input_images - 1)
        
    def forward(
        self, 
        target_img: torch.Tensor, 
        source_imgs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Predict pose from target to each source image.
        
        Args:
            target_img: (B, 3, H, W) target/reference image
            source_imgs: List of (B, 3, H, W) source images
            
        Returns:
            poses: List of (B, 4, 4) transformation matrices
        """
        # Concatenate target with each source and predict pose
        poses = []
        
        for source_img in source_imgs:
            # Concatenate target and source
            inputs = torch.cat([target_img, source_img], dim=1)
            
            # Encode
            features = self.encoder(inputs)
            
            # Decode to pose parameters
            axisangle, translation = self.decoder(features)
            
            # Convert to transformation matrix
            T = transformation_from_parameters(
                axisangle.squeeze(1), 
                translation.squeeze(1)
            )
            poses.append(T)
        
        return poses


class PoseNetShared(nn.Module):
    """
    PoseNet with shared encoder for efficiency.
    Processes all frames together.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Shared encoder (single image)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        
        # Pose decoder (takes concatenated features)
        self.pose_conv = nn.Sequential(
            nn.Conv2d(512 * 2, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 6, 1)
        )
        
    def forward(
        self, 
        target_img: torch.Tensor, 
        source_imgs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        # Encode target
        target_feat = self.encoder(target_img)
        
        poses = []
        for source_img in source_imgs:
            # Encode source
            source_feat = self.encoder(source_img)
            
            # Concatenate features
            combined = torch.cat([target_feat, source_feat], dim=1)
            
            # Predict pose
            x = self.pose_conv(combined)
            x = x.mean(dim=[2, 3])
            x = 0.01 * x
            
            axisangle = x[:, :3]
            translation = x[:, 3:]
            
            T = transformation_from_parameters(axisangle, translation)
            poses.append(T)
        
        return poses
