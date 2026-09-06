"""Contrast Limited Adaptive Histogram Equalization (CLAHE) for fundus images."""

from typing import Tuple
import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to enhance local contrast in the Lightness channel of an RGB fundus image.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an HxWxC 3-channel NumPy array.")
    if image.dtype != np.uint8:
        raise ValueError(f"Input image must have dtype uint8, got {image.dtype}")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be a positive float, got {clip_limit}")
    if (
        not isinstance(tile_grid_size, (tuple, list))
        or len(tile_grid_size) != 2
        or tile_grid_size[0] <= 0
        or tile_grid_size[1] <= 0
    ):
        raise ValueError(f"tile_grid_size must be a tuple of 2 positive integers, got {tile_grid_size}")

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size[0]), int(tile_grid_size[1])),
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)