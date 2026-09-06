from typing import Dict, Any, Optional
import numpy as np

def compute_saliency_mask_metrics(
    heatmap: np.ndarray,
    lesion_mask: Optional[np.ndarray],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes localization metrics against binary lesion masks if available.
    Distinguishes saliency localization from true segmentation.
    """
    if lesion_mask is None or np.sum(lesion_mask) == 0:
        return {
            "lesion_overlap_ratio": None,
            "pointing_game_hit": None,
            "mean_lesion_energy": None,
            "note": "Lesion annotations not available or no lesion present."
        }

    total_saliency = float(np.sum(heatmap))
    if total_saliency < 1e-8:
        return {"lesion_overlap_ratio": 0.0, "pointing_game_hit": 0, "mean_lesion_energy": 0.0}

    lesion_binary = (lesion_mask > 0).astype(np.float32)
    saliency_in_lesion = float(np.sum(heatmap * lesion_binary))
    lesion_overlap_ratio = saliency_in_lesion / total_saliency

    # Pointing Game: is peak attribution located inside a true lesion?
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    pointing_hit = int(lesion_binary[max_idx[0], max_idx[1]] > 0)

    mean_energy = float(np.mean(heatmap[lesion_binary > 0]))

    return {
        "lesion_overlap_ratio": round(lesion_overlap_ratio, 4),
        "pointing_game_hit": pointing_hit,
        "mean_lesion_energy": round(mean_energy, 4),
    }
