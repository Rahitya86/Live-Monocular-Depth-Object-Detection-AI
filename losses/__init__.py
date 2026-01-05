# Losses module
"""Loss functions for self-supervised depth training."""

from .ssim import SSIM, SSIMLoss, compute_ssim_loss
from .photometric import (
    PhotometricLoss,
    ReprojectionLoss,
    compute_photometric_loss,
    compute_auto_mask,
    compute_reprojection_loss,
    compute_min_reprojection_loss
)
from .smoothness import (
    SmoothnessLoss,
    compute_smoothness_loss,
    compute_second_order_smoothness,
    compute_multiscale_smoothness,
    get_scaled_images
)
from .combined import MonoDepthLoss, MonoDepthLossV2, compute_depth_metrics
from .temporal import (
    TemporalConsistencyLoss,
    FlowConsistencyLoss,
    DepthConsistencyLoss
)
from .stereo import (
    StereoConsistencyLoss,
    LRConsistencyLoss,
    DisparitySmoothnessLoss
)

__all__ = [
    'SSIM',
    'SSIMLoss',
    'compute_ssim_loss',
    'PhotometricLoss',
    'ReprojectionLoss',
    'compute_photometric_loss',
    'compute_auto_mask',
    'compute_reprojection_loss',
    'compute_min_reprojection_loss',
    'SmoothnessLoss',
    'compute_smoothness_loss',
    'compute_second_order_smoothness',
    'compute_multiscale_smoothness',
    'get_scaled_images',
    'MonoDepthLoss',
    'MonoDepthLossV2',
    'compute_depth_metrics',
    'TemporalConsistencyLoss',
    'FlowConsistencyLoss',
    'DepthConsistencyLoss',
    'StereoConsistencyLoss',
    'LRConsistencyLoss',
    'DisparitySmoothnessLoss'
]
