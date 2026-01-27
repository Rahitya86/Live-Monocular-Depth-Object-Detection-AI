# Assets Folder

This folder contains visual assets for the README and documentation.

## Required Images

Add the following images to make the README visually compelling:

### Hero Banner
- `hero-banner.png` - Main hero image showing the product in action (recommended: 1920x600px)

### Demo Screenshots
- `demo-screenshot.png` - Screenshot of live inference window
- `demo-outdoor.gif` - GIF showing outdoor depth estimation (driving scene)
- `demo-indoor.gif` - GIF showing indoor depth estimation (room/office)
- `depth-colormap.png` - Side-by-side comparison of different colormaps
- `object-detection.png` - Screenshot showing object detection with distance labels

### Industry Icons (64x64px each)
- `icon-robotics.png` - Robot icon
- `icon-security.png` - Security/camera icon
- `icon-arvr.png` - VR headset icon
- `icon-automotive.png` - Car icon
- `icon-retail.png` - Shopping cart icon
- `icon-drone.png` - Drone icon
- `icon-healthcare.png` - Medical cross icon
- `icon-industrial.png` - Factory/gear icon

## Generating Demo Assets

### Record a Demo GIF

```bash
# 1. Run inference and save frames
python inference_live.py --webcam 0 --checkpoint checkpoints/best.pth

# 2. Press 'S' to save frames of interest

# 3. Convert to GIF using ffmpeg or online tools
ffmpeg -framerate 10 -pattern_type glob -i 'saved_frame_*.png' -vf "scale=640:-1" demo.gif
```

### Create Colormap Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a synthetic depth map
depth = np.random.rand(192, 640)

colormaps = ['magma', 'viridis', 'plasma', 'turbo']
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for ax, cmap in zip(axes.flat, colormaps):
    ax.imshow(depth, cmap=cmap)
    ax.set_title(cmap)
    ax.axis('off')
plt.savefig('depth-colormap.png', dpi=150, bbox_inches='tight')
```

## Free Icon Resources

- [Heroicons](https://heroicons.com/) - MIT licensed
- [Feather Icons](https://feathericons.com/) - MIT licensed
- [Lucide Icons](https://lucide.dev/) - ISC licensed
- [Font Awesome](https://fontawesome.com/) - Various licenses
