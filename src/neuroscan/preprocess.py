"""Morphological skull stripping and extreme-point contour cropping pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from neuroscan.models import BoundingBox

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def load_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """Load image from path, PIL image, or array into RGB numpy array."""
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at: {path}")
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            # Grayscale to RGB
            return np.stack([image_input] * 3, axis=-1)
        elif image_input.shape[2] == 4:
            # RGBA to RGB
            return image_input[:, :, :3]
        return image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def crop_brain_contour(
    image_rgb: np.ndarray,
    threshold_val: int = 45,
) -> Tuple[np.ndarray, BoundingBox]:
    """
    Isolates the brain parenchyma and eliminates surrounding black canvas / skull bone
    by finding the largest external contour and its extreme spatial boundaries.
    
    Returns:
        (cropped_image_rgb, bounding_box)
    """
    h, w = image_rgb.shape[:2]

    if HAS_OPENCV:
        # Convert RGB to Grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Apply Gaussian Blur to smooth noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Segment tissue via thresholding
        _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
        # Morphological erosion and dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, kernel, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Select contour with largest area (the brain skull perimeter)
            c = max(contours, key=cv2.contourArea)
            min_x = int(c[:, :, 0].min())
            max_x = int(c[:, :, 0].max())
            min_y = int(c[:, :, 1].min())
            max_y = int(c[:, :, 1].max())

            # Ensure non-empty crop
            if max_x > min_x and max_y > min_y:
                cropped = image_rgb[min_y:max_y, min_x:max_x]
                return cropped, BoundingBox(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

    # Fallback to NumPy bounding box if OpenCV is absent or contour extraction fails
    gray = np.mean(image_rgb, axis=2)
    mask = gray > threshold_val
    if np.any(mask):
        y_indices, x_indices = np.where(mask)
        min_y, max_y = int(y_indices.min()), int(y_indices.max())
        min_x, max_x = int(x_indices.min()), int(x_indices.max())
        if max_x > min_x and max_y > min_y:
            return (
                image_rgb[min_y : max_y + 1, min_x : max_x + 1],
                BoundingBox(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y),
            )

    # Default to entire image if no foreground detected
    return image_rgb, BoundingBox(min_x=0, max_x=w, min_y=0, max_y=h)


def preprocess_mri(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    apply_crop: bool = True,
) -> Tuple[np.ndarray, np.ndarray, BoundingBox]:
    """
    End-to-end preprocessing pipeline for VGG-16 inference:
    1. Load image
    2. Contour extreme-point crop (skull stripping)
    3. Resize to target dimension (e.g. 224x224)
    4. Normalize pixel tensor to [0, 1] range

    Returns:
        (normalized_tensor, cropped_rgb_image, bounding_box)
    """
    image_rgb = load_image(image_input)

    if apply_crop:
        cropped_rgb, bbox = crop_brain_contour(image_rgb)
    else:
        cropped_rgb = image_rgb
        h, w = image_rgb.shape[:2]
        bbox = BoundingBox(min_x=0, max_x=w, min_y=0, max_y=h)

    # Resize using high-quality Lanczos interpolation
    pil_cropped = Image.fromarray(cropped_rgb)
    pil_resized = pil_cropped.resize(target_size, Image.Resampling.LANCZOS)
    resized_rgb = np.array(pil_resized)

    # Normalize float32 tensor
    tensor = resized_rgb.astype(np.float32) / 255.0
    return tensor, resized_rgb, bbox
