import numpy as np
import pytest

from neuroscan.gradcam import (
    apply_colormap_jet,
    compute_gradcam_weights,
    overlay_heatmap_on_image,
)


def test_apply_colormap_jet():
    heatmap = np.linspace(0.0, 1.0, 100).reshape((10, 10))
    colored = apply_colormap_jet(heatmap)

    assert colored.shape == (10, 10, 3)
    assert colored.dtype == np.uint8


def test_overlay_heatmap_on_image():
    base_img = np.full((50, 50, 3), 100, dtype=np.uint8)
    heatmap = np.ones((50, 50), dtype=np.float32)

    overlay = overlay_heatmap_on_image(base_img, heatmap, alpha=0.5)
    assert overlay.shape == (50, 50, 3)
    assert overlay.dtype == np.uint8


def test_compute_gradcam_weights():
    # 7x7 spatial resolution with 4 channels
    activations = np.ones((7, 7, 4), dtype=np.float32)
    # Channel 0 has strong positive gradients, other channels zero
    gradients = np.zeros((7, 7, 4), dtype=np.float32)
    gradients[:, :, 0] = 1.0

    cam = compute_gradcam_weights(activations, gradients)
    assert cam.shape == (7, 7)
    assert 0.0 <= np.min(cam) <= np.max(cam) <= 1.0
