"""
DepthNet: Complete depth estimation network.
Modern architecture with encoder-decoder + attention.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .encoder import get_encoder
from .decoder import DepthDecoder, LightweightDecoder, disp_to_depth


class DepthNet(nn.Module):
    """
    Complete depth estimation network.
    
    Supports multiple encoder architectures:
    - ResNet (18/34/50/101)
    - EfficientNet (b0/b3/b5)
    - ConvNeXt (tiny/small/base)
    - DPT/ViT (vitb16/vitl16)
    - Hybrid (CNN + Transformer)
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        pretrained: bool = True,
        scales: List[int] = [0, 1, 2, 3],
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        use_attention: bool = True,
        lightweight: bool = False
    ):
        super().__init__()
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scales = scales
        self.encoder_name = encoder_name
        
        # Encoder
        self.encoder = get_encoder(encoder_name, pretrained)
        
        # Decoder - choose based on memory requirements
        if lightweight or not use_attention:
            self.decoder = LightweightDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                scales=scales
            )
        else:
            self.decoder = DepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                scales=scales,
                use_attention=use_attention
            )
        
    def forward(self, x: torch.Tensor) -> Dict:
        """
        Predict depth from single image.
        
        Args:
            x: (B, 3, H, W) input image, normalized to ImageNet stats
            
        Returns:
            outputs: Dict containing:
                - ('disp', scale): Disparity at each scale
                - ('depth', scale): Depth at each scale
        """
        # Encode
        features = self.encoder(x)
        
        # Decode to disparity
        outputs = self.decoder(features)
        
        # Convert disparity to depth
        for scale in self.scales:
            if ('disp', scale) in outputs:
                disp = outputs[('disp', scale)]
                depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                outputs[('depth', scale)] = depth
            
        return outputs
    
    def get_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Get single depth output at highest resolution."""
        outputs = self.forward(x)
        return outputs[('depth', 0)]
