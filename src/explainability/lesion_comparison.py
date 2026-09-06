"""Exploratory evaluation of Grad-CAM attention against IDRiD pixel lesion masks."""
from typing import Dict
import numpy as np


def evaluate_saliency_pointing_game(heatmap: np.ndarray, lesion_mask: np.ndarray) -> Dict[str, float]:
    """Compute Pointing Game Hit Rate evaluating if peak attention hits true lesion areas."""
    raise NotImplementedError("Lesion saliency evaluation is scheduled for Phase 9.")