from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_cam_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Blends a normalized [0, 1] 2D heatmap over an RGB image uint8 in [0, 255].
    Returns blended RGB image as uint8.
    """
    H, W = image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color, alpha, 0)
    return blended

def extract_retinal_mask(image_rgb: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Extracts binary mask (1 inside circular retinal fundus, 0 outside black border).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY)
    # Morphological closing to remove small interior holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def compute_attention_distribution(
    heatmap: np.ndarray,
    image_rgb: np.ndarray
) -> Tuple[float, float]:
    """
    Calculates proportion of saliency mass inside the retinal field vs outer black borders/artifacts.
    """
    mask = extract_retinal_mask(image_rgb)
    total_saliency = float(np.sum(heatmap))
    if total_saliency < 1e-8:
        return 0.0, 0.0

    retinal_saliency = float(np.sum(heatmap * mask))
    background_saliency = float(np.sum(heatmap * (1 - mask)))

    retinal_ratio = retinal_saliency / total_saliency
    border_ratio = background_saliency / total_saliency
    return retinal_ratio, border_ratio
