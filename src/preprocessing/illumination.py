"""Local color subtraction for fundus illumination normalization."""

import cv2
import numpy as np


def normalize_illumination(
    image: np.ndarray,
    sigma: float = 10.0,
) -> np.ndarray:
    """
    Reduce smooth illumination variation using local color subtraction.

    A Gaussian-smoothed version of each channel is used as an estimate
    of slowly varying illumination. The local component is then centered
    around the original channel mean.

    Parameters
    ----------
    image:
        RGB uint8 image.
    sigma:
        Gaussian blur standard deviation.

    Returns
    -------
    np.ndarray
        RGB uint8 illumination-normalized image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    image_float = image.astype(np.float32)

    result = np.empty_like(image_float)

    for channel in range(3):
        original = image_float[:, :, channel]

        blurred = cv2.GaussianBlur(
            original,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        normalized = original - blurred + float(original.mean())

        result[:, :, channel] = normalized

    return np.clip(result, 0, 255).astype(np.uint8)
