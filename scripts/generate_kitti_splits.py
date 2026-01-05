#!/usr/bin/env python3
"""
KITTI Raw Dataset File List Generator.

Scans the KITTI Raw dataset directory and generates train/val/test splits
for monocular depth estimation training.

Features:
- Automatically discovers all drives
- Generates Eigen split compatible files
- Supports both image_02 (left) and image_03 (right) cameras
- Creates proper temporal triplets

Usage:
    python scripts/generate_kitti_splits.py --data_root /path/to/kitti_raw --output_dir ./splits
"""

import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


# KITTI recording dates
KITTI_DATES = [
    '2011_09_26',
    '2011_09_28', 
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]

# Eigen test split scenes (excluded from training)
EIGEN_TEST_SCENES = [
    '2011_09_26_drive_0002_sync',
    '2011_09_26_drive_0005_sync',
    '2011_09_26_drive_0009_sync',
    '2011_09_26_drive_0013_sync',
    '2011_09_26_drive_0020_sync',
    '2011_09_26_drive_0023_sync',
    '2011_09_26_drive_0027_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0036_sync',
    '2011_09_26_drive_0046_sync',
    '2011_09_26_drive_0048_sync',
    '2011_09_26_drive_0052_sync',
    '2011_09_26_drive_0056_sync',
    '2011_09_26_drive_0059_sync',
    '2011_09_26_drive_0060_sync',
    '2011_09_26_drive_0064_sync',
    '2011_09_26_drive_0084_sync',
    '2011_09_26_drive_0086_sync',
    '2011_09_26_drive_0091_sync',
    '2011_09_26_drive_0093_sync',
    '2011_09_26_drive_0096_sync',
    '2011_09_26_drive_0101_sync',
    '2011_09_26_drive_0104_sync',
    '2011_09_26_drive_0106_sync',
    '2011_09_26_drive_0113_sync',
    '2011_09_26_drive_0117_sync',
    '2011_09_28_drive_0001_sync',
    '2011_09_28_drive_0002_sync',
    '2011_09_29_drive_0026_sync',
    '2011_09_29_drive_0071_sync',
]


def discover_drives(data_root: Path) -> Dict[str, List[Path]]:
    """
    Discover all KITTI drives.
    
    Returns:
        Dict mapping date -> list of drive paths
    """
    drives = defaultdict(list)
    
    for date in KITTI_DATES:
        date_dir = data_root / date
        if not date_dir.exists():
            continue
        
        for item in sorted(date_dir.iterdir()):
            if item.is_dir() and 'drive' in item.name and '_sync' in item.name:
                # Check if it has images
                img_dir = item / 'image_02' / 'data'
                if img_dir.exists():
                    drives[date].append(item)
    
    return drives


def count_frames(drive_path: Path, cam: str = 'image_02') -> int:
    """Count frames in a drive."""
    img_dir = drive_path / cam / 'data'
    if not img_dir.exists():
        return 0
    
    # Count PNG and JPG files
    pngs = list(img_dir.glob('*.png'))
    jpgs = list(img_dir.glob('*.jpg'))
    
    return len(pngs) + len(jpgs)


def generate_samples(
    drives: Dict[str, List[Path]],
    frame_ids: List[int] = [0, -1, 1],
    exclude_scenes: List[str] = None
) -> List[Tuple[str, int, str]]:
    """
    Generate all valid samples.
    
    Args:
        drives: Dict of date -> drive paths
        frame_ids: Frame offsets for temporal context
        exclude_scenes: Scene names to exclude
        
    Returns:
        List of (folder, frame_idx, side) tuples
    """
    samples = []
    exclude_scenes = exclude_scenes or []
    
    min_offset = min(frame_ids)
    max_offset = max(frame_ids)
    
    for date, drive_list in drives.items():
        for drive_path in drive_list:
            # Check if excluded
            if drive_path.name in exclude_scenes:
                continue
            
            num_frames = count_frames(drive_path)
            if num_frames == 0:
                continue
            
            # Generate samples
            folder = f"{date}/{drive_path.name}"
            
            for idx in range(abs(min_offset), num_frames - max_offset):
                # Left camera
                samples.append((folder, idx, 'l'))
                # Right camera (for stereo)
                samples.append((folder, idx, 'r'))
    
    return samples


def generate_eigen_test_samples(
    drives: Dict[str, List[Path]],
    test_scenes: List[str]
) -> List[Tuple[str, int, str]]:
    """Generate Eigen test split samples."""
    samples = []
    
    for date, drive_list in drives.items():
        for drive_path in drive_list:
            if drive_path.name not in test_scenes:
                continue
            
            num_frames = count_frames(drive_path)
            folder = f"{date}/{drive_path.name}"
            
            # Sample every 10th frame for test
            for idx in range(0, num_frames, 10):
                samples.append((folder, idx, 'l'))
    
    return samples


def write_file_list(samples: List[Tuple[str, int, str]], output_path: Path):
    """Write samples to file."""
    with open(output_path, 'w') as f:
        for folder, idx, side in samples:
            f.write(f"{folder} {idx:010d} {side}\n")
    
    print(f"Written {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate KITTI file lists')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to KITTI raw data root')
    parser.add_argument('--output_dir', type=str, default='./splits',
                        help='Output directory for file lists')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--include_stereo', action='store_true',
                        help='Include right camera samples')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning KITTI data in: {data_root}")
    
    # Discover drives
    drives = discover_drives(data_root)
    
    total_drives = sum(len(d) for d in drives.values())
    print(f"Found {total_drives} drives across {len(drives)} dates")
    
    for date, drive_list in drives.items():
        print(f"  {date}: {len(drive_list)} drives")
    
    # Generate train samples (excluding Eigen test scenes)
    print("\nGenerating training samples...")
    train_samples = generate_samples(
        drives, 
        frame_ids=[0, -1, 1],
        exclude_scenes=EIGEN_TEST_SCENES
    )
    
    if not args.include_stereo:
        train_samples = [(f, i, s) for f, i, s in train_samples if s == 'l']
    
    # Shuffle and split
    random.shuffle(train_samples)
    
    n_val = int(len(train_samples) * args.val_ratio)
    val_samples = train_samples[:n_val]
    train_samples = train_samples[n_val:]
    
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Generate Eigen test samples
    print("\nGenerating Eigen test samples...")
    test_samples = generate_eigen_test_samples(drives, EIGEN_TEST_SCENES)
    print(f"Test samples: {len(test_samples)}")
    
    # Write files
    print("\nWriting file lists...")
    write_file_list(train_samples, output_dir / 'kitti_train_files.txt')
    write_file_list(val_samples, output_dir / 'kitti_val_files.txt')
    write_file_list(test_samples, output_dir / 'kitti_test_eigen.txt')
    
    # Also write a combined file
    all_samples = train_samples + val_samples
    random.shuffle(all_samples)
    write_file_list(all_samples, output_dir / 'kitti_all_train.txt')
    
    # Write summary
    summary = {
        'data_root': str(data_root),
        'total_drives': total_drives,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'dates': list(drives.keys()),
        'excluded_scenes': EIGEN_TEST_SCENES
    }
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nDone!")
    print(f"\nTo train with full KITTI dataset:")
    print(f"  python train_advanced.py --data_root {data_root} --epochs 25")


if __name__ == '__main__':
    main()
