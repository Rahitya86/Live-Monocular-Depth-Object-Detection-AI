"""
Modern depth decoder with attention and feature fusion.
Produces multi-scale disparity maps with skip connections.

Optimizations for 90-95% accuracy:
- Feature Pyramid Network (FPN) style fusion
- Attention-based feature refinement
- Proper upsampling with learned weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ConvBlock(nn.Module):
    """Conv + BatchNorm + ELU block."""
    
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        return self.elu(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    """Bilinear upsample + Conv block."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.elu(self.conv(x))


class AttentionBlock(nn.Module):
    """Channel attention for feature refinement."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class FeatureFusionBlock(nn.Module):
    """Feature Pyramid Network style fusion block."""
    
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.project = nn.Conv2d(in_ch, out_ch, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        self.attention = AttentionBlock(out_ch) if use_attention else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        x = self.project(x)
        if skip is not None:
            # Resize x to match skip if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = x + skip
        x = self.relu(x + self.refine(x))
        x = self.attention(x)
        return x


class DepthDecoder(nn.Module):
    """
    Modern depth decoder with FPN-style feature fusion.
    
    Key improvements:
    - Feature pyramid network fusion
    - Channel attention
    - Multi-scale disparity output
    - Proper gradient flow with residual connections
    """
    
    def __init__(
        self,
        num_ch_enc: List[int],
        scales: List[int] = [0, 1, 2, 3],
        num_output_channels: int = 1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_output_channels = num_output_channels
        
        # Decoder channels (FPN style - all same dimension)
        self.fpn_dim = 256
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ch, self.fpn_dim, 1) for ch in num_ch_enc
        ])
        
        # Top-down pathway with fusion
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(self.fpn_dim, self.fpn_dim, use_attention)
            for _ in range(len(num_ch_enc))
        ])
        
        # Disparity output heads for each scale
        self.disp_heads = nn.ModuleDict()
        for s in scales:
            self.disp_heads[f'disp_{s}'] = nn.Sequential(
                nn.Conv2d(self.fpn_dim, 128, 3, padding=1),
                nn.ELU(inplace=True),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.ELU(inplace=True),
                nn.Conv2d(64, num_output_channels, 3, padding=1),
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, encoder_features: List[torch.Tensor]) -> Dict:
        """
        Decode features to multi-scale disparity maps.
        """
        outputs = {}
        
        # Apply lateral convolutions
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, encoder_features)]
        
        # Top-down pathway (from deepest to shallowest)
        fpn_features = [None] * len(laterals)
        fpn_features[-1] = self.fusion_blocks[-1](laterals[-1])
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample deeper feature
            upsampled = F.interpolate(
                fpn_features[i + 1], 
                size=laterals[i].shape[2:],
                mode='bilinear', 
                align_corners=True
            )
            # Fuse with lateral
            fpn_features[i] = self.fusion_blocks[i](laterals[i] + upsampled)
        
        # Generate disparity outputs at each scale
        for scale in self.scales:
            # Map scale to feature index
            feat_idx = min(scale, len(fpn_features) - 1)
            feat = fpn_features[feat_idx]
            
            # Generate disparity
            disp = self.disp_heads[f'disp_{scale}'](feat)
            outputs[('disp', scale)] = disp
            
        return outputs


class LightweightDecoder(nn.Module):
    """
    Lightweight decoder for faster inference.
    Uses simple upsampling with skip connections.
    """
    
    def __init__(
        self,
        num_ch_enc: List[int],
        scales: List[int] = [0, 1, 2, 3],
        num_output_channels: int = 1
    ):
        super().__init__()
        
        self.scales = scales
        self.num_ch_dec = [16, 32, 64, 128, 256]
        
        self.convs = nn.ModuleDict()
        
        for i in range(4, -1, -1):
            num_ch_in = num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f'upconv_{i}_0'] = UpConvBlock(num_ch_in, num_ch_out)
            
            if i > 0:
                num_ch_in = num_ch_out + num_ch_enc[i - 1]
            else:
                num_ch_in = num_ch_out
            self.convs[f'upconv_{i}_1'] = ConvBlock(num_ch_in, num_ch_out)
            
        for s in scales:
            self.convs[f'dispconv_{s}'] = nn.Conv2d(self.num_ch_dec[s], num_output_channels, 3, padding=1)
            
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, encoder_features: List[torch.Tensor]) -> Dict:
        outputs = {}
        x = encoder_features[-1]
        
        for i in range(4, -1, -1):
            x = self.convs[f'upconv_{i}_0'](x)
            if i > 0:
                x = torch.cat([x, encoder_features[i - 1]], dim=1)
            x = self.convs[f'upconv_{i}_1'](x)
            
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[f'dispconv_{i}'](x))
                
        return outputs


def disp_to_depth(disp: torch.Tensor, min_depth: float = 0.1, max_depth: float = 100.0):
    """Convert disparity to depth using inverse relationship."""
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return depth
