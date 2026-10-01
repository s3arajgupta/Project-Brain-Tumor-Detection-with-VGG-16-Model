import numpy as np
import pytest
from PIL import Image

from neuroscan.preprocess import crop_brain_contour, load_image, preprocess_mri


def test_load_image_numpy():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    loaded = load_image(arr)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)


def test_crop_brain_contour_synthetic():
    # Create black canvas 200x200 with bright circle in center (representing skull/brain)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    y, x = np.ogrid[:200, :200]
    mask = (x - 100) ** 2 + (y - 100) ** 2 < 40 ** 2
    img[mask] = 200

    cropped, bbox = crop_brain_contour(img, threshold_val=45)
    assert cropped.shape[0] < 200
    assert cropped.shape[1] < 200
    assert bbox.width > 0
    assert bbox.height > 0
    assert bbox.min_x >= 50
    assert bbox.max_x <= 150


def test_preprocess_mri_pipeline():
    img = np.full((150, 150, 3), 128, dtype=np.uint8)
    tensor, resized, bbox = preprocess_mri(img, target_size=(224, 224))

    assert tensor.shape == (224, 224, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0
    assert resized.shape == (224, 224, 3)
