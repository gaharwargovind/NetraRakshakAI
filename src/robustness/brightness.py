"""Exposure variation corruption generators."""
import numpy as np


def apply_exposure_corruption(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply underexposure or overexposure scaling."""
    raise NotImplementedError("Exposure corruption is scheduled for Phase 11.")