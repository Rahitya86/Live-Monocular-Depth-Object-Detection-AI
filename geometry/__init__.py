# Geometry module for self-supervised depth estimation
"""
Differentiable geometry operations for monocular depth estimation.
Includes camera intrinsics, projection/backprojection, and inverse warping.
"""

from .camera import (
    get_intrinsics,
    get_scaled_intrinsics,
    pixel2cam,
    cam2pixel,
    normalize_coords
)

from .transform import (
    euler_to_rotation_matrix,
    axisangle_to_rotation_matrix,
    pose_vec_to_matrix,
    invert_pose
)

from .warping import (
    inverse_warp,
    compute_projection_coords
)

__all__ = [
    'get_intrinsics',
    'get_scaled_intrinsics',
    'pixel2cam',
    'cam2pixel',
    'normalize_coords',
    'euler_to_rotation_matrix',
    'axisangle_to_rotation_matrix',
    'pose_vec_to_matrix',
    'invert_pose',
    'inverse_warp',
    'compute_projection_coords'
]
