"""
Unit and integration tests for the SIH26038 evaluation framework.
Verifies metrics, confusion matrices, ROC-AUC, error handling, and E001 reproduction.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.confusion_matrix import compute_confusion_matrices
from src.evaluation.metrics import (
    compute_clinical_screening_metrics,
    compute_five_class_metrics,
    compute_probability_metrics,
    compute_referable_dr_metrics,
    evaluate_predictions,
    validate_evaluation_inputs,
)
from src.evaluation.roc import compute_referable_roc_curve, compute_roc_pr_curves


def test_perfect_predictions():
    """1. Test that exact predictions produce 1.0 across all metrics."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    y_prob = np.eye(5)

    res = evaluate_predictions(y_true, y_pred, y_prob)
    assert res["five_class"]["accuracy"] == 1.0
    assert res["five_class"]["macro_f1"] == 1.0
    assert res["five_class"]["quadratic_weighted_kappa"] == 1.0
    assert res["referable_dr"]["sensitivity"] == 1.0
    assert res["referable_dr"]["specificity"] == 1.0
    assert res["probability_metrics"]["referable_roc_auc"] == 1.0


def test_completely_wrong_predictions():
    """2. Test opposite predictions to verify degradation response."""
    y_true = np.array([0, 0, 0, 4, 4])
    y_pred = np.array([4, 4, 4, 0, 0])
    res = evaluate_predictions(y_true, y_pred)
    assert res["five_class"]["accuracy"] == 0.0
    assert res["five_class"]["macro_f1"] == 0.0
    assert res["referable_dr"]["sensitivity"] == 0.0
    assert res["referable_dr"]["specificity"] == 0.0


def test_imbalanced_class_distributions():
    """3. Test handling of severe class imbalance (e.g. rare Grade 3/4 cases)."""
    y_true = np.array([0] * 50 + [1] * 20 + [2] * 15 + [3] * 3 + [4] * 2)
    y_pred = y_true.copy()
    y_pred[-1] = 0  # Miss one Grade 4 case
    res = evaluate_predictions(y_true, y_pred)
    assert res["five_class"]["per_class"]["4"]["recall"] == 0.5
    assert res["five_class"]["accuracy"] > 0.95


def test_missing_class_in_predictions():
    """4. Test that zero predicted samples for a class handled with zero-division safety."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 2, 2])  # Never predicts 3 or 4
    res = evaluate_predictions(y_true, y_pred)
    assert res["five_class"]["per_class"]["3"]["precision"] == 0.0
    assert res["five_class"]["per_class"]["3"]["recall"] == 0.0
    assert res["five_class"]["per_class"]["4"]["precision"] == 0.0


def test_referable_conversion():
    """5. Test exact threshold mapping where Grade >= 2 is referable."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    ref = compute_referable_dr_metrics(y_true, y_pred, referable_threshold=2)
    assert ref["counts"]["tn"] == 2  # Grades 0, 1
    assert ref["counts"]["tp"] == 3  # Grades 2, 3, 4
    assert ref["counts"]["fp"] == 0
    assert ref["counts"]["fn"] == 0


def test_sensitivity_calculation():
    """6. Test sensitivity formula: TP / (TP + FN)."""
    # 4 True referables (2, 2, 3, 4), model predicts referable for 3 of them
    y_true = np.array([2, 2, 3, 4])
    y_pred = np.array([2, 0, 3, 4])  # One FN (predicted 0)
    ref = compute_referable_dr_metrics(y_true, y_pred, referable_threshold=2)
    assert np.isclose(ref["sensitivity"], 3.0 / 4.0)


def test_specificity_calculation():
    """7. Test specificity formula: TN / (TN + FP)."""
    # 4 True non-referables (0, 0, 1, 1), model predicts non-referable for 3 of them
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 2])  # One FP (predicted 2)
    ref = compute_referable_dr_metrics(y_true, y_pred, referable_threshold=2)
    assert np.isclose(ref["specificity"], 3.0 / 4.0)


def test_macro_f1():
    """8. Test unweighted macro F1 across 5 classes."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 0])  # Misses class 4
    m = compute_five_class_metrics(y_true, y_pred)
    # classes 0-3 have F1=1.0, 1.0, 1.0, 1.0, class 4 has F1=0.0, class 0 has F1=0.667
    assert 0.0 < m["macro_f1"] < 1.0


def test_qwk_distance_penalty():
    """9. Test that distant errors are penalized more heavily by QWK."""
    y_true = np.array([0, 0, 0, 0])
    y_close = np.array([1, 1, 1, 1])  # Off by 1 grade
    y_distant = np.array([4, 4, 4, 4])  # Off by 4 grades

    # Use reference labels [0..4] to define standard 5-class matrix bounds
    qwk_close = float(compute_five_class_metrics(np.concatenate([y_true, [4]]), np.concatenate([y_close, [4]]))["quadratic_weighted_kappa"])
    qwk_dist = float(compute_five_class_metrics(np.concatenate([y_true, [4]]), np.concatenate([y_distant, [4]]))["quadratic_weighted_kappa"])
    assert qwk_close > qwk_dist


def test_confusion_matrices():
    """10 & 11. Test raw count and row-normalized confusion matrix shapes and sums."""
    y_true = np.array([0, 0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 1, 2, 3, 4])
    raw_cm, norm_cm = compute_confusion_matrices(y_true, y_pred, num_classes=5)
    assert raw_cm.shape == (5, 5)
    assert norm_cm.shape == (5, 5)
    # Row 0 had 2 samples: 1 predicted 0, 1 predicted 1 -> normalized row: [0.5, 0.5, 0, 0, 0]
    assert np.allclose(norm_cm[0], [0.5, 0.5, 0.0, 0.0, 0.0])


def test_roc_auc_metrics():
    """12. Test ROC-AUC calculation for binary and one-vs-rest distributions."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_prob = np.eye(5)
    res = compute_probability_metrics(y_true, y_prob, referable_threshold=2)
    assert res["referable_roc_auc"] == 1.0
    assert res["multiclass_roc_auc_ovr_macro"] == 1.0


def test_invalid_label_handling():
    """13. Test exception raising when labels exceed valid bounds."""
    with pytest.raises(ValueError, match="outside \\[0, 4\\]"):
        validate_evaluation_inputs(y_true=[0, 1, 5], y_pred=[0, 1, 2])

    with pytest.raises(ValueError, match="outside \\[0, 4\\]"):
        validate_evaluation_inputs(y_true=[0, 1, 2], y_pred=[-1, 1, 2])


def test_invalid_probability_shape():
    """14. Test exception raising when probability matrix dimensions are invalid."""
    with pytest.raises(ValueError, match="must have shape"):
        validate_evaluation_inputs(y_true=[0, 1], y_pred=[0, 1], y_prob=[[0.5, 0.5], [0.5, 0.5]])


def test_probability_validation_sum():
    """15. Test rejection of unnormalized probability vectors."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_evaluation_inputs(
            y_true=[0, 1],
            y_pred=[0, 1],
            y_prob=[[0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]],
        )


def test_reproduce_e001_baseline_metrics():
    """
    16. Verification test: Confirm the evaluation framework reproduces the
    exact test-set metrics recorded in experiments/E001_baseline/metrics.json.
    """
    metrics_path = Path("experiments/E001_baseline/metrics.json")
    assert metrics_path.is_file(), "E001 metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        stored_data = json.load(f)

    test_metrics = stored_data["held_out_test_metrics"]
    cm = np.array(test_metrics["confusion_matrix"])

    # Reconstruct exact true and predicted class pairs from the recorded confusion matrix
    y_true_list = []
    y_pred_list = []
    for i in range(5):
        for j in range(5):
            count = int(cm[i, j])
            y_true_list.extend([i] * count)
            y_pred_list.extend([j] * count)

    y_true = np.array(y_true_list, dtype=int)
    y_pred = np.array(y_pred_list, dtype=int)

    recomputed = evaluate_predictions(y_true=y_true, y_pred=y_pred)

    # Verify metric consistency against stored test metrics
    assert np.isclose(recomputed["five_class"]["accuracy"], test_metrics["accuracy"], atol=1e-3)
    assert np.isclose(recomputed["five_class"]["macro_f1"], test_metrics["macro_f1"], atol=1e-3)
    assert np.isclose(recomputed["five_class"]["quadratic_weighted_kappa"], test_metrics["quadratic_weighted_kappa"], atol=1e-3)
    assert np.isclose(recomputed["referable_dr"]["sensitivity"], test_metrics["referable_sensitivity"], atol=1e-3)
    assert np.isclose(recomputed["referable_dr"]["specificity"], test_metrics["referable_specificity"], atol=1e-3)