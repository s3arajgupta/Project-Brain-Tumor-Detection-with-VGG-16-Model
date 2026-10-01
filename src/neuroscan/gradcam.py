"""Gradient-weighted Class Activation Mapping (Grad-CAM) visual explainability engine."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def apply_colormap_jet(heatmap: np.ndarray) -> np.ndarray:
    """
    Applies a Jet colormap (Blue -> Cyan -> Yellow -> Red) to a 2D [0, 1] heatmap.
    Returns RGB uint8 image of shape (H, W, 3).
    """
    if HAS_OPENCV:
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Pure NumPy implementation of Jet colormap
    x = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.uint8(255 * rgb)


def overlay_heatmap_on_image(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto the original RGB MRI scan.
    
    Args:
        original_rgb: (H, W, 3) uint8 image.
        heatmap: (H, W) float array normalized between 0.0 and 1.0.
        alpha: Blending weight for heatmap overlay (0.0 to 1.0).
    """
    h, w = original_rgb.shape[:2]

    # Resize heatmap to match original image dimensions if needed
    if heatmap.shape[:2] != (h, w):
        pil_heat = Image.fromarray(heatmap.astype(np.float32))
        pil_heat_resized = pil_heat.resize((w, h), Image.Resampling.BILINEAR)
        heatmap = np.array(pil_heat_resized)

    # Generate colored heatmap
    colored_heatmap = apply_colormap_jet(heatmap)

    # Blend original and colored heatmap
    blended = (1.0 - alpha) * original_rgb.astype(np.float32) + alpha * colored_heatmap.astype(np.float32)
    return np.uint8(np.clip(blended, 0, 255))


def compute_gradcam_weights(
    activations: np.ndarray,
    gradients: np.ndarray,
) -> np.ndarray:
    """
    Computes 2D Grad-CAM heatmap from convolutional feature activations and gradients.
    
    Args:
        activations: (H, W, Channels) feature map activations.
        gradients: (H, W, Channels) gradients of score w.r.t activations.
    
    Returns:
        (H, W) normalized heatmap in [0.0, 1.0].
    """
    # Global average pooling over spatial dimensions to get channel importance weights
    weights = np.mean(gradients, axis=(0, 1))

    # Weighted linear combination of forward activation maps
    cam = np.zeros(activations.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[:, :, i]

    # Apply ReLU: only positive correlations contribute to the class of interest
    cam = np.maximum(cam, 0)

    # Min-max normalization
    cam_max = np.max(cam)
    if cam_max > 0:
        cam /= cam_max

    return cam
