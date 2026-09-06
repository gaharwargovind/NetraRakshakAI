# FILE: src/preprocessing/crop_retina.py
"""Retinal field cropping and black border removal for fundus images."""

from typing import Tuple
import cv2
import numpy as np


def crop_retina_circle(
    image: np.ndarray,
    threshold_val: int = 10,
    margin_pct: float = 0.01,
) -> np.ndarray:
    """
    Isolate the active retinal circle and crop outer bounding black borders.

    Args:
        image: RGB fundus image as uint8 NumPy array (H, W, 3).
        threshold_val: Grayscale intensity cutoff to segment foreground retina.
        margin_pct: Safety boundary margin added around detected bounding box.

    Returns:
        Cropped RGB NumPy array. If bounding box cannot be resolved, returns original image.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Input image must be an HxWxC 3-channel NumPy array.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold_val

    if not np.any(mask):
        return image

    y_indices, x_indices = np.where(mask)
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))

    h, w, _ = image.shape
    margin_y = int(h * margin_pct)
    margin_x = int(w * margin_pct)

    y_min = max(0, y_min - margin_y)
    y_max = min(h, y_max + margin_y)
    x_min = max(0, x_min - margin_x)
    x_max = min(w, x_max + margin_x)

    if (y_max - y_min) < 32 or (x_max - x_min) < 32:
        return image

    return image[y_min:y_max, x_min:x_max]