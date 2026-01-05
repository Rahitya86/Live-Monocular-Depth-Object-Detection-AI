"""
Modern encoder architectures for depth estimation.
Supports: ResNet, EfficientNet, ConvNeXt, and DPT (Vision Transformer).

Key improvements for 90-95% accuracy:
- DPT encoder for state-of-the-art accuracy
- EfficientNet for better efficiency
- ConvNeXt for modern CNN performance
- Proper feature pyramid extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List


class ResNetEncoder(nn.Module):
    """ResNet-based encoder with multi-scale feature extraction."""
    
    def __init__(self, num_layers: int = 50, pretrained: bool = True):
        super().__init__()
        self.num_layers = num_layers
        
        if num_layers == 18:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            encoder = models.resnet18(weights=weights)
            self.num_ch_enc = [64, 64, 128, 256, 512]
        elif num_layers == 34:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            encoder = models.resnet34(weights=weights)
            self.num_ch_enc = [64, 64, 128, 256, 512]
        elif num_layers == 50:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            encoder = models.resnet50(weights=weights)
            self.num_ch_enc = [64, 256, 512, 1024, 2048]
        elif num_layers == 101:
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            encoder = models.resnet101(weights=weights)
            self.num_ch_enc = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported num_layers: {num_layers}")
        
        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x = self.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


class EfficientNetEncoder(nn.Module):
    """EfficientNet encoder for better efficiency."""
    
    def __init__(self, variant: str = 'b5', pretrained: bool = True):
        super().__init__()
        
        if variant == 'b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b0(weights=weights)
            self.num_ch_enc = [16, 24, 40, 112, 320]
        elif variant == 'b3':
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b3(weights=weights)
            self.num_ch_enc = [24, 32, 48, 136, 384]
        elif variant == 'b5':
            weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b5(weights=weights)
            self.num_ch_enc = [24, 40, 64, 176, 512]
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        self.features = efficientnet.features
        self.extract_indices = [1, 2, 3, 5, 8]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.extract_indices:
                features.append(x)
        while len(features) < 5:
            features.append(features[-1])
        return features[:5]


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt encoder - modern CNN architecture."""
    
    def __init__(self, variant: str = 'base', pretrained: bool = True):
        super().__init__()
        
        if variant == 'tiny':
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            convnext = models.convnext_tiny(weights=weights)
            self.num_ch_enc = [96, 96, 192, 384, 768]
        elif variant == 'small':
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            convnext = models.convnext_small(weights=weights)
            self.num_ch_enc = [96, 96, 192, 384, 768]
        elif variant == 'base':
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            convnext = models.convnext_base(weights=weights)
            self.num_ch_enc = [128, 128, 256, 512, 1024]
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        self.stem = convnext.features[0]
        self.stage1 = convnext.features[1]
        self.stage2 = nn.Sequential(convnext.features[2], convnext.features[3])
        self.stage3 = nn.Sequential(convnext.features[4], convnext.features[5])
        self.stage4 = nn.Sequential(convnext.features[6], convnext.features[7])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        x0 = self.stem(x)
        features.append(x0)
        x1 = self.stage1(x0)
        features.append(x1)
        x2 = self.stage2(x1)
        features.append(x2)
        x3 = self.stage3(x2)
        features.append(x3)
        x4 = self.stage4(x3)
        features.append(x4)
        return features


class DPTEncoder(nn.Module):
    """
    DPT (Dense Prediction Transformer) encoder.
    Uses Vision Transformer backbone - state-of-the-art for depth estimation.
    """
    
    def __init__(self, variant: str = 'vitb16', pretrained: bool = True):
        super().__init__()
        
        if variant == 'vitb16':
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            vit = models.vit_b_16(weights=weights)
            self.embed_dim = 768
            self.patch_size = 16
            self.num_ch_enc = [256, 512, 768, 768, 768]
            self.num_layers = 12
        elif variant == 'vitl16':
            weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            vit = models.vit_l_16(weights=weights)
            self.embed_dim = 1024
            self.patch_size = 16
            self.num_ch_enc = [256, 512, 1024, 1024, 1024]
            self.num_layers = 24
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding
        self.blocks = vit.encoder.layers
        self.norm = vit.encoder.ln
        
        # Readout projections
        self.readout_projs = nn.ModuleList([
            nn.Conv2d(self.embed_dim, ch, 1) for ch in self.num_ch_enc
        ])
        
        # Hook indices
        n = len(self.blocks)
        self.hook_indices = [n//4, n//2, 3*n//4, n-1]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token and pos embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        if x.shape[1] != self.pos_embed.shape[1]:
            pos_embed = self._interpolate_pos_embed(self.pos_embed, h, w)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        
        # Extract intermediate features
        intermediate = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.hook_indices:
                intermediate.append(x)
        
        x = self.norm(x)
        intermediate.append(x)
        
        # Convert to spatial features
        features = []
        for idx, (feat, proj) in enumerate(zip(intermediate, self.readout_projs)):
            spatial = feat[:, 1:, :].transpose(1, 2).view(B, self.embed_dim, h, w)
            spatial = proj(spatial)
            scale_factor = 2 ** max(0, 3 - idx)
            if scale_factor > 1:
                spatial = F.interpolate(spatial, scale_factor=scale_factor, 
                                       mode='bilinear', align_corners=True)
            features.append(spatial)
        
        return features
    
    def _interpolate_pos_embed(self, pos_embed, h, w):
        N = pos_embed.shape[1] - 1
        cls_pos = pos_embed[:, :1]
        patch_pos = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]
        orig_size = int(N ** 0.5)
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bilinear', align_corners=True)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)


class HybridEncoder(nn.Module):
    """Hybrid CNN-Transformer encoder for best accuracy/efficiency trade-off."""
    
    def __init__(self, cnn_backbone: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        
        num_layers = int(cnn_backbone.replace('resnet', ''))
        self.cnn = ResNetEncoder(num_layers=num_layers, pretrained=pretrained)
        
        self.embed_dim = 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.proj_in = nn.Conv2d(self.cnn.num_ch_enc[-1], self.embed_dim, 1)
        self.proj_out = nn.Conv2d(self.embed_dim, self.cnn.num_ch_enc[-1], 1)
        self.num_ch_enc = self.cnn.num_ch_enc
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.cnn(x)
        deep_feat = features[-1]
        B, C, H, W = deep_feat.shape
        
        deep_feat = self.proj_in(deep_feat)
        deep_flat = deep_feat.flatten(2).transpose(1, 2)
        deep_flat = self.transformer(deep_flat)
        deep_feat = deep_flat.transpose(1, 2).view(B, self.embed_dim, H, W)
        deep_feat = self.proj_out(deep_feat)
        
        features[-1] = deep_feat
        return features


def get_encoder(name: str = 'resnet50', pretrained: bool = True) -> nn.Module:
    """
    Factory function for encoders.
    
    Supported:
    - ResNet: 'resnet18/34/50/101'
    - EfficientNet: 'efficientnet_b0/b3/b5'
    - ConvNeXt: 'convnext_tiny/small/base'
    - DPT/ViT: 'dpt_vitb16/vitl16'
    - Hybrid: 'hybrid_resnet50'
    """
    name = name.lower()
    
    if name.startswith('resnet'):
        num_layers = int(name.replace('resnet', ''))
        return ResNetEncoder(num_layers=num_layers, pretrained=pretrained)
    elif name.startswith('efficientnet'):
        variant = name.replace('efficientnet_', '')
        return EfficientNetEncoder(variant=variant, pretrained=pretrained)
    elif name.startswith('convnext'):
        variant = name.replace('convnext_', '')
        return ConvNeXtEncoder(variant=variant, pretrained=pretrained)
    elif name.startswith('dpt') or name.startswith('vit'):
        variant = name.replace('dpt_', '').replace('vit_', '')
        if variant in ['vitb16', 'b16', 'base']:
            variant = 'vitb16'
        elif variant in ['vitl16', 'l16', 'large']:
            variant = 'vitl16'
        return DPTEncoder(variant=variant, pretrained=pretrained)
    elif name.startswith('hybrid'):
        cnn = name.replace('hybrid_', '') or 'resnet50'
        return HybridEncoder(cnn_backbone=cnn, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown encoder: {name}")
