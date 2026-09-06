"""
Confusion matrix generation and visualization for 5-class ICDR and Referable DR.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from src.evaluation.metrics import ICDR_CLASS_NAMES


def compute_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raw count and row-normalized confusion matrices.
    """
    labels = list(range(num_classes))
    raw_cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = raw_cm.sum(axis=1, keepdims=True).astype(float)
    norm_cm = np.divide(raw_cm, row_sums, out=np.zeros_like(raw_cm, dtype=float), where=row_sums > 0)
    return raw_cm, norm_cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render and optionally save a styled confusion matrix plot.
    """
    if class_names is None:
        class_names = ICDR_CLASS_NAMES[: cm.shape[0]]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    default_title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix (Counts)"
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title or default_title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            val_str = f"{val:.2f}" if normalize else str(int(val))
            ax.text(
                j,
                i,
                val_str,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


def save_confusion_matrix_plot(
    cm: Union[List[List[int]], np.ndarray],
    class_names: List[str],
    output_path: Union[str, Path],
    normalize: bool = False,
) -> None:
    """
    Backwards-compatible wrapper preserving E001 training script interface.
    """
    cm_arr = np.asarray(cm)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True).astype(float)
        cm_arr = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums > 0)
    fig = plot_confusion_matrix(cm_arr, class_names=class_names, normalize=normalize, output_path=output_path)
    plt.close(fig)