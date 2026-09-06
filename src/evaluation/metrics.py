"""
Centralized Clinical & Machine Learning Evaluation Framework.
SIH Problem Statement: SIH26038 — Explainable AI for DR Screening in Rural India.

Provides standard evaluation for:
1. Five-class ICDR disease severity classification
2. Binary Referable DR screening triage (ICDR Grade >= 2)
3. Probabilistic metrics (One-vs-Rest and Referable ROC-AUC)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

ICDR_CLASS_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


def validate_evaluation_inputs(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Optional[Union[np.ndarray, List[int]]] = None,
    y_prob: Optional[Union[np.ndarray, List[List[float]]]] = None,
    num_classes: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Validate and standardize evaluation arrays.
    
    Args:
        y_true: Ground truth class integers in [0, num_classes - 1].
        y_pred: Predicted class integers in [0, num_classes - 1]. If None, derived from argmax(y_prob).
        y_prob: Predicted probabilities of shape (N, num_classes). Optional.
        num_classes: Expected number of classes (default: 5).

    Returns:
        Tuple of (y_true, y_pred, y_prob) as standardized NumPy arrays.

    Raises:
        ValueError: If array dimensions, shapes, label ranges, or probability constraints are violated.
    """
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be a 1-dimensional array, got shape {y_true_arr.shape}")

    n_samples = len(y_true_arr)
    if n_samples == 0:
        raise ValueError("Cannot evaluate empty input arrays (n_samples == 0)")

    if not np.issubdtype(y_true_arr.dtype, np.integer):
        raise ValueError(f"y_true must contain integer class labels, got dtype {y_true_arr.dtype}")

    if np.any(y_true_arr < 0) or np.any(y_true_arr >= num_classes):
        invalid_vals = y_true_arr[(y_true_arr < 0) | (y_true_arr >= num_classes)]
        raise ValueError(f"y_true contains labels outside [0, {num_classes - 1}]: {np.unique(invalid_vals)}")

    # Validate y_prob if provided
    y_prob_arr: Optional[np.ndarray] = None
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        if y_prob_arr.ndim != 2 or y_prob_arr.shape != (n_samples, num_classes):
            raise ValueError(
                f"y_prob must have shape ({n_samples}, {num_classes}), got {y_prob_arr.shape}"
            )
        if np.any(np.isnan(y_prob_arr)) or np.any(np.isinf(y_prob_arr)):
            raise ValueError("y_prob contains NaN or infinite values")
        if np.any(y_prob_arr < -1e-6) or np.any(y_prob_arr > 1.0 + 1e-6):
            raise ValueError("y_prob values must lie within [0.0, 1.0]")
        row_sums = np.sum(y_prob_arr, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-2):
            raise ValueError(f"y_prob rows must sum to 1.0. Found row sums min={row_sums.min():.4f}, max={row_sums.max():.4f}")

    # Validate or derive y_pred
    if y_pred is None:
        if y_prob_arr is None:
            raise ValueError("Either y_pred or y_prob must be provided for evaluation")
        y_pred_arr = np.argmax(y_prob_arr, axis=1).astype(int)
    else:
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim != 1:
            raise ValueError(f"y_pred must be a 1-dimensional array, got shape {y_pred_arr.shape}")
        if len(y_pred_arr) != n_samples:
            raise ValueError(f"Length mismatch: len(y_true)={n_samples} != len(y_pred)={len(y_pred_arr)}")
        if not np.issubdtype(y_pred_arr.dtype, np.integer):
            raise ValueError(f"y_pred must contain integer labels, got dtype {y_pred_arr.dtype}")
        if np.any(y_pred_arr < 0) or np.any(y_pred_arr >= num_classes):
            invalid_vals = y_pred_arr[(y_pred_arr < 0) | (y_pred_arr >= num_classes)]
            raise ValueError(f"y_pred contains labels outside [0, {num_classes - 1}]: {np.unique(invalid_vals)}")

    return y_true_arr, y_pred_arr, y_prob_arr


def compute_five_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Calculate 5-class ICDR classification performance metrics.
    """
    labels = list(range(num_classes))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0.0))
    qwk = float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic"))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0.0
    )

    per_class = {}
    for c in labels:
        per_class[str(c)] = {
            "name": ICDR_CLASS_NAMES[c] if c < len(ICDR_CLASS_NAMES) else f"Class_{c}",
            "precision": float(precisions[c]),
            "recall": float(recalls[c]),
            "f1": float(f1s[c]),
            "support": int(supports[c]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Normalized confusion matrix (row-normalized by true support)
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    norm_cm_arr = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums > 0)
    normalized_cm = norm_cm_arr.tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "quadratic_weighted_kappa": qwk,
        "per_class": per_class,
        "confusion_matrix": cm,
        "normalized_confusion_matrix": normalized_cm,
    }


def compute_referable_dr_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    referable_threshold: int = 2,
) -> Dict[str, Any]:
    """
    Calculate binary Referable DR screening metrics.
    Positive: ICDR Grade >= referable_threshold (Default: 2 - Moderate NPDR, Severe NPDR, PDR)
    Negative: ICDR Grade < referable_threshold (Default: Grade 0 - No DR, Grade 1 - Mild NPDR)
    """
    y_true_bin = (y_true >= referable_threshold).astype(int)
    y_pred_bin = (y_pred >= referable_threshold).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2 * (precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    return {
        "referable_threshold": referable_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "counts": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def compute_probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    referable_threshold: int = 2,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Calculate probability-based metrics including One-vs-Rest and Referable ROC-AUC.
    """
    results: Dict[str, Any] = {
        "multiclass_roc_auc_ovr_macro": None,
        "multiclass_roc_auc_ovr_weighted": None,
        "per_class_roc_auc": {},
        "referable_roc_auc": None,
    }

    # Per-class One-vs-Rest ROC-AUC
    valid_aucs = []
    weights = []
    for c in range(num_classes):
        y_c = (y_true == c).astype(int)
        prob_c = y_prob[:, c]
        if len(np.unique(y_c)) == 2:
            auc_c = float(roc_auc_score(y_c, prob_c))
            results["per_class_roc_auc"][str(c)] = auc_c
            valid_aucs.append(auc_c)
            weights.append(np.sum(y_c))
        else:
            results["per_class_roc_auc"][str(c)] = None

    if len(valid_aucs) == num_classes:
        results["multiclass_roc_auc_ovr_macro"] = float(np.mean(valid_aucs))
        results["multiclass_roc_auc_ovr_weighted"] = float(np.average(valid_aucs, weights=weights))
    elif len(valid_aucs) > 0:
        results["multiclass_roc_auc_ovr_macro"] = float(np.mean(valid_aucs))

    # Binary Referable ROC-AUC (Sum of posterior probabilities for grades >= referable_threshold)
    y_true_bin = (y_true >= referable_threshold).astype(int)
    prob_referable = np.sum(y_prob[:, referable_threshold:], axis=1)

    if len(np.unique(y_true_bin)) == 2:
        results["referable_roc_auc"] = float(roc_auc_score(y_true_bin, prob_referable))
    else:
        results["referable_roc_auc"] = None

    return results


def evaluate_predictions(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Optional[Union[np.ndarray, List[int]]] = None,
    y_prob: Optional[Union[np.ndarray, List[List[float]]]] = None,
    referable_threshold: int = 2,
    num_classes: int = 5,
) -> Dict[str, Any]:
    """
    Unified evaluation entry-point returning complete structured metrics.
    """
    y_true_arr, y_pred_arr, y_prob_arr = validate_evaluation_inputs(
        y_true, y_pred, y_prob, num_classes=num_classes
    )

    five_class_metrics = compute_five_class_metrics(y_true_arr, y_pred_arr, num_classes=num_classes)
    referable_metrics = compute_referable_dr_metrics(
        y_true_arr, y_pred_arr, referable_threshold=referable_threshold
    )

    payload: Dict[str, Any] = {
        "n_samples": int(len(y_true_arr)),
        "five_class": five_class_metrics,
        "referable_dr": referable_metrics,
    }

    if y_prob_arr is not None:
        payload["probability_metrics"] = compute_probability_metrics(
            y_true_arr, y_prob_arr, referable_threshold=referable_threshold, num_classes=num_classes
        )

    return payload


def compute_clinical_screening_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred_probs: Union[np.ndarray, List[List[float]]],
    y_pred: Optional[Union[np.ndarray, List[int]]] = None,
    referable_threshold: int = 2,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute clinical screening metrics with backward-compatible support for referable_threshold."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_pred_probs, dtype=np.float32)

    if y_prob_arr.ndim == 1:
        if y_pred is None:
            y_pred_arr = y_prob_arr.astype(np.int64)
        else:
            y_pred_arr = np.asarray(y_pred, dtype=np.int64)
        y_prob_arr = np.eye(5, dtype=np.float32)[y_pred_arr]
    else:
        if y_pred is None:
            y_pred_arr = np.argmax(y_prob_arr, axis=1)
        else:
            y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    full_eval = evaluate_predictions(
        y_true_arr,
        y_pred_arr,
        y_prob_arr,
        referable_threshold=referable_threshold,
        **kwargs,
    )

    return {
        "accuracy": float(full_eval["five_class"]["accuracy"]),
        "macro_f1": float(full_eval["five_class"]["macro_f1"]),
        "quadratic_weighted_kappa": float(full_eval["five_class"]["quadratic_weighted_kappa"]),
        "referable_sensitivity": float(full_eval["referable_dr"]["sensitivity"]),
        "referable_specificity": float(full_eval["referable_dr"]["specificity"]),
        "confusion_matrix": full_eval["five_class"]["confusion_matrix"],
        "per_class": full_eval["five_class"]["per_class"],
        "five_class": full_eval["five_class"],
        "referable_dr": full_eval["referable_dr"],
        "probability_metrics": full_eval.get("probability_metrics", {}),
    }

def decode_ordinal_probabilities(
    probabilities: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Decode discrete grade by threshold summation: sum_{k=1}^4 I(p_k > threshold)."""
    return (probabilities > threshold).sum(dim=-1).to(torch.long)


def reconstruct_ordinal_class_probabilities(
    cumulative_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct discrete class probability mass distribution from ordered cumulative probabilities:
        P(y = 0) = 1 - p1
        P(y = 1) = p1 - p2
        P(y = 2) = p2 - p3
        P(y = 3) = p3 - p4
        P(y = 4) = p4
    """
    p1 = cumulative_probs[..., 0:1]
    p2 = cumulative_probs[..., 1:2]
    p3 = cumulative_probs[..., 2:3]
    p4 = cumulative_probs[..., 3:4]

    p0 = 1.0 - p1
    p_class1 = p1 - p2
    p_class2 = p2 - p3
    p_class3 = p3 - p4
    p_class4 = p4

    return torch.cat([p0, p_class1, p_class2, p_class3, p_class4], dim=-1)


def compute_ordinal_error_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE, adjacent error rate, and severe error rate (|y - y_hat| >= 2)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    errors = np.abs(y_true - y_pred)
    n = len(y_true)

    return {
        "mean_absolute_error": float(np.mean(errors)),
        "exact_agreement_rate": float(np.mean(errors == 0)),
        "adjacent_error_rate": float(np.mean(errors == 1)),
        "severe_error_rate": float(np.mean(errors >= 2)),
    }