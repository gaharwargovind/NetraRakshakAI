"""
Receiver Operating Characteristic (ROC) analysis and visual curve plotting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from src.evaluation.metrics import ICDR_CLASS_NAMES


def compute_roc_pr_curves(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute binary ROC and PR coordinates and AUC scores.
    """
    y_true_bin = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    if len(np.unique(y_true_bin)) < 2:
        return {
            "roc_auc": None,
            "pr_auc": None,
            "fpr": [],
            "tpr": [],
            "precision": [],
            "recall": [],
        }

    fpr, tpr, roc_thresholds = roc_curve(y_true_bin, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true_bin, scores)

    return {
        "roc_auc": float(auc(fpr, tpr)),
        "pr_auc": float(auc(recall, precision)),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "pr_thresholds": pr_thresholds.tolist(),
    }


def compute_multiclass_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute One-vs-Rest ROC curves across all classes.
    """
    num_classes = y_prob.shape[1]
    if class_names is None:
        class_names = ICDR_CLASS_NAMES[:num_classes]

    curves: Dict[str, Any] = {}
    for c in range(num_classes):
        y_c = (y_true == c).astype(int)
        scores_c = y_prob[:, c]
        curve_data = compute_roc_pr_curves(y_c, scores_c)
        curves[class_names[c]] = curve_data
    return curves


def compute_referable_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referable_threshold: int = 2,
) -> Dict[str, Any]:
    """
    Compute ROC curve for Referable DR screening (ICDR Grade >= referable_threshold).
    """
    y_true_ref = (y_true >= referable_threshold).astype(int)
    prob_ref = np.sum(y_prob[:, referable_threshold:], axis=1)
    return compute_roc_pr_curves(y_true_ref, prob_ref)


def plot_roc_curves(
    curves_data: Dict[str, Any],
    title: str = "ROC Curves",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render and optionally save ROC curves.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for name, data in curves_data.items():
        if data.get("roc_auc") is not None:
            fpr = data["fpr"]
            tpr = data["tpr"]
            auc_val = data["roc_auc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig