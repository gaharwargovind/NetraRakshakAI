import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

from src.losses.focal_loss import MulticlassFocalLoss
from src.losses.class_balanced_focal import ClassBalancedFocalLoss

def test_alpha_ones_matches_e007_focal_loss():
    """When alpha=[1,1,1,1,1], ClassBalancedFocalLoss must match MulticlassFocalLoss identically."""
    cb_fl = ClassBalancedFocalLoss(gamma=2.0, alpha=[1.0, 1.0, 1.0, 1.0, 1.0], reduction="mean")
    e007_fl = MulticlassFocalLoss(gamma=2.0, reduction="mean")

    logits = torch.randn(20, 5, requires_grad=True)
    targets = torch.randint(0, 5, (20,))

    loss_cb = cb_fl(logits, targets)
    loss_e007 = e007_fl(logits, targets)

    assert torch.allclose(loss_cb, loss_e007, atol=1e-6)

def test_per_class_alpha_scales_loss_expectedly():
    """Scaling alpha for a specific class must scale its loss contribution proportionately."""
    alpha = [1.0, 2.5, 1.0, 1.0, 1.0]
    cb_fl = ClassBalancedFocalLoss(gamma=2.0, alpha=alpha, reduction="none")
    unweighted_fl = ClassBalancedFocalLoss(gamma=2.0, alpha=None, reduction="none")

    logits = torch.randn(5, 5)
    targets = torch.tensor([0, 1, 2, 3, 4])

    loss_cb = cb_fl(logits, targets)
    loss_unweighted = unweighted_fl(logits, targets)

    # Class 1 must be scaled by exactly 2.5
    assert torch.isclose(loss_cb[1], loss_unweighted[1] * 2.5, atol=1e-5)
    # Class 0 must remain unscaled
    assert torch.isclose(loss_cb[0], loss_unweighted[0], atol=1e-5)

def test_gradients_are_finite_and_stable():
    """Gradients must remain finite under extreme prediction margins."""
    alpha = [0.5, 2.0, 1.0, 3.0, 2.5]
    cb_fl = ClassBalancedFocalLoss(gamma=2.0, alpha=alpha, reduction="mean")

    logits = torch.tensor([[-50.0, 50.0, 0.0, 10.0, -10.0]], requires_grad=True)
    targets = torch.tensor([1])

    loss = cb_fl(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.all(torch.isfinite(logits.grad))

def test_validation_cohort_count_invariance():
    """Validation cohort must remain strictly 601 records."""
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    assert val_csv.is_file()
    df = pd.read_csv(val_csv)
    assert len(df) == 601

def test_test_set_isolation():
    """Held-out test partition must not be referenced by the training script."""
    train_script = ROOT_DIR / "scripts/train_e009.py"
    if train_script.is_file():
        text = train_script.read_text()
        assert "data/processed/aptos/test.csv" not in text
        assert "test.csv" not in text
