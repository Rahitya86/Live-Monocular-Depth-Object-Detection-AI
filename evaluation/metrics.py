"""
KITTI Eigen split evaluation metrics.

Standard metrics for monocular depth estimation:
- Abs Rel: Absolute relative error
- Sq Rel: Squared relative error  
- RMSE: Root mean squared error
- RMSE log: RMSE of log depth
- δ < 1.25, 1.25², 1.25³: Threshold accuracies

Target for 90-95% accuracy: δ < 1.25 = 0.90-0.95
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class DepthMetrics:
    """
    Compute standard depth estimation metrics.
    
    Metrics computed:
    - abs_rel: Mean absolute relative error
    - sq_rel: Mean squared relative error
    - rmse: Root mean squared error
    - rmse_log: RMSE of log depth
    - a1: % of pixels with δ < 1.25
    - a2: % of pixels with δ < 1.25²
    - a3: % of pixels with δ < 1.25³
    """
    
    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        use_median_scaling: bool = True,
        garg_crop: bool = True
    ):
        """
        Args:
            min_depth: Minimum depth for evaluation
            max_depth: Maximum depth for evaluation (80m for KITTI)
            use_median_scaling: Use median scaling for scale alignment
            garg_crop: Use Garg/Eigen crop for KITTI evaluation
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_median_scaling = use_median_scaling
        self.garg_crop = garg_crop
        
        # Accumulate metrics
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.metrics = {
            'abs_rel': [],
            'sq_rel': [],
            'rmse': [],
            'rmse_log': [],
            'a1': [],
            'a2': [],
            'a3': []
        }
        self.count = 0
    
    def compute_metrics(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute depth metrics for a single sample.
        
        Args:
            pred: (H, W) predicted depth
            gt: (H, W) ground truth depth
            
        Returns:
            metrics: Dict of metric values
        """
        # Apply Garg/Eigen crop if specified
        if self.garg_crop:
            pred, gt = self._apply_garg_crop(pred, gt)
        
        # Create valid mask
        mask = (gt > self.min_depth) & (gt < self.max_depth)
        
        if mask.sum() < 100:
            return None
        
        pred = pred[mask]
        gt = gt[mask]
        
        # Median scaling (align scale)
        if self.use_median_scaling:
            ratio = np.median(gt) / np.median(pred)
            pred = pred * ratio
        
        # Clip predictions
        pred = np.clip(pred, self.min_depth, self.max_depth)
        
        # Compute errors
        thresh = np.maximum(gt / pred, pred / gt)
        
        metrics = {
            'abs_rel': np.mean(np.abs(gt - pred) / gt),
            'sq_rel': np.mean(((gt - pred) ** 2) / gt),
            'rmse': np.sqrt(np.mean((gt - pred) ** 2)),
            'rmse_log': np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2)),
            'a1': np.mean(thresh < 1.25),
            'a2': np.mean(thresh < 1.25 ** 2),
            'a3': np.mean(thresh < 1.25 ** 3)
        }
        
        return metrics
    
    def update(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ):
        """Add a sample to accumulated metrics."""
        sample_metrics = self.compute_metrics(pred, gt)
        
        if sample_metrics is not None:
            for key in self.metrics:
                self.metrics[key].append(sample_metrics[key])
            self.count += 1
    
    def get_results(self) -> Dict[str, float]:
        """Get mean metrics over all samples."""
        if self.count == 0:
            return {k: 0.0 for k in self.metrics}
        
        return {k: np.mean(v) for k, v in self.metrics.items()}
    
    def _apply_garg_crop(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Garg/Eigen crop for KITTI evaluation."""
        H, W = gt.shape
        
        # Garg crop parameters
        top = int(0.40810811 * H)
        left = int(0.03594771 * W)
        bottom = int(0.99189189 * H)
        right = int(0.96405229 * W)
        
        pred = pred[top:bottom, left:right]
        gt = gt[top:bottom, left:right]
        
        return pred, gt


def evaluate_kitti(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on KITTI Eigen split.
    
    Args:
        model: Depth estimation model
        dataloader: DataLoader with GT depth
        device: Computation device
        max_samples: Maximum samples to evaluate
        verbose: Print progress
        
    Returns:
        metrics: Dict of evaluation metrics
    """
    model.eval()
    
    metrics = DepthMetrics(
        min_depth=0.001,
        max_depth=80.0,
        use_median_scaling=True,
        garg_crop=True
    )
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_samples and i >= max_samples:
                break
            
            # Get input image
            input_img = batch[('color', 0, 0)].to(device)
            
            # Get GT depth
            if 'depth_gt' not in batch:
                continue
            gt_depth = batch['depth_gt'].numpy()
            
            # Predict depth
            outputs = model(input_img)
            pred_depth = outputs[('depth', 0)].cpu().numpy()
            
            # Update metrics for each sample in batch
            B = pred_depth.shape[0]
            for b in range(B):
                pred = pred_depth[b, 0]
                gt = gt_depth[b, 0]
                
                # Resize pred to match GT if needed
                if pred.shape != gt.shape:
                    import cv2
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                
                metrics.update(pred, gt)
            
            if verbose and (i + 1) % 10 == 0:
                current = metrics.get_results()
                print(f"Sample {i+1}: δ<1.25={current['a1']:.3f}, "
                      f"AbsRel={current['abs_rel']:.3f}")
    
    results = metrics.get_results()
    
    if verbose:
        print("\n" + "=" * 60)
        print("KITTI Eigen Split Evaluation Results")
        print("=" * 60)
        print(f"  Abs Rel: {results['abs_rel']:.4f}")
        print(f"  Sq Rel:  {results['sq_rel']:.4f}")
        print(f"  RMSE:    {results['rmse']:.4f}")
        print(f"  RMSE log:{results['rmse_log']:.4f}")
        print(f"  δ < 1.25:   {results['a1']:.4f} ({results['a1']*100:.1f}%)")
        print(f"  δ < 1.25²:  {results['a2']:.4f} ({results['a2']*100:.1f}%)")
        print(f"  δ < 1.25³:  {results['a3']:.4f} ({results['a3']*100:.1f}%)")
        print("=" * 60)
    
    return results


def compute_depth_errors(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None
) -> Dict[str, torch.Tensor]:
    """
    Compute depth errors for a batch (PyTorch version).
    
    Args:
        pred: (B, 1, H, W) predicted depth
        gt: (B, 1, H, W) ground truth depth
        mask: (B, 1, H, W) valid mask
        
    Returns:
        errors: Dict of error tensors
    """
    if mask is None:
        mask = (gt > 0.001) & (gt < 80.0)
    
    # Flatten
    pred = pred[mask]
    gt = gt[mask]
    
    if pred.numel() == 0:
        return {
            'abs_rel': torch.tensor(0.0),
            'sq_rel': torch.tensor(0.0),
            'rmse': torch.tensor(0.0),
            'a1': torch.tensor(0.0)
        }
    
    # Median scaling
    ratio = torch.median(gt) / torch.median(pred)
    pred = pred * ratio
    
    # Clip
    pred = torch.clamp(pred, 0.001, 80.0)
    
    # Errors
    thresh = torch.maximum(gt / pred, pred / gt)
    
    errors = {
        'abs_rel': torch.mean(torch.abs(gt - pred) / gt),
        'sq_rel': torch.mean(((gt - pred) ** 2) / gt),
        'rmse': torch.sqrt(torch.mean((gt - pred) ** 2)),
        'a1': torch.mean((thresh < 1.25).float())
    }
    
    return errors


class EigenSplitLoader:
    """
    Load KITTI Eigen split test files.
    """
    
    # Eigen test files (697 images)
    EIGEN_TEST_FILES = [
        "2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png",
        "2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000054.png",
        # ... (full list would be loaded from file)
    ]
    
    @staticmethod
    def load_eigen_test_files(filepath: str) -> List[str]:
        """Load test file list."""
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    @staticmethod
    def get_gt_depth_path(image_path: str, gt_root: str) -> str:
        """Get corresponding GT depth path."""
        # KITTI depth format
        parts = image_path.split('/')
        date = parts[0]
        drive = parts[1]
        frame = parts[-1].replace('.png', '.png').replace('.jpg', '.png')
        
        return f"{gt_root}/{date}/{drive}/proj_depth/groundtruth/image_02/{frame}"
