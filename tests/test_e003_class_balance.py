"""Unit and contract tests for E003 class-weight calculation, weighted loss, and isolation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.data.loaders import create_e003_data_loaders
from src.training.losses import compute_class_weights, get_loss_function
from src.utils.config import load_config


def test_compute_class_weights_formula():
    """Verify standard formula N / (K * N_c) on synthetic integer labels."""
    synthetic_labels = [0] * 100 + [1] * 50 + [2] * 25 + [3] * 15 + [4] * 10
    weights = compute_class_weights(synthetic_labels, num_classes=5)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (5,)
    assert weights.dtype == torch.float32

    expected_0 = 200.0 / (5 * 100)  # 0.4
    expected_1 = 200.0 / (5 * 50)   # 0.8
    expected_2 = 200.0 / (5 * 25)   # 1.6
    expected_3 = 200.0 / (5 * 15)   # 2.6667
    expected_4 = 200.0 / (5 * 10)   # 4.0

    assert np.isclose(weights[0].item(), expected_0)
    assert np.isclose(weights[1].item(), expected_1)
    assert np.isclose(weights[2].item(), expected_2)
    assert np.isclose(weights[3].item(), expected_3)
    assert np.isclose(weights[4].item(), expected_4)


def test_compute_class_weights_properties_on_train_split():
    """Verify class weights computed on actual train.csv are positive, finite, and ordered."""
    train_csv = Path("data/processed/aptos/train.csv")
    assert train_csv.is_file(), "train.csv must exist"
    df = pd.read_csv(train_csv)
    if "is_conflicting_duplicate" in df.columns:
        df = df[df["is_conflicting_duplicate"] != True]

    weights = compute_class_weights(df["diagnosis"].values, num_classes=5)

    assert weights.shape == (5,)
    assert torch.all(weights > 0), "All class weights must be strictly positive"
    assert torch.all(torch.isfinite(weights)), "All class weights must be finite"

    # Minority classes (3, 4, 1) must have higher weight than majority class 0
    assert weights[3] > weights[0]
    assert weights[4] > weights[0]
    assert weights[1] > weights[0]


def test_compute_class_weights_error_handling():
    """Verify validation of invalid labels or missing classes."""
    with pytest.raises(ValueError, match="non-empty 1D array"):
        compute_class_weights([])

    with pytest.raises(ValueError, match="0 samples detected"):
        compute_class_weights([0, 1, 2, 3], num_classes=5)

    with pytest.raises(ValueError, match="within \\[0, 4\\]"):
        compute_class_weights([0, 1, 2, 5], num_classes=5)


def test_weighted_cross_entropy_loss_behavior():
    """
    Verify Weighted Cross-Entropy applies higher penalty to minority-class errors.
    Uses a 2-sample batch to evaluate relative weighting under default reduction='mean'.
    """
    weights = torch.tensor([0.4, 2.0, 0.7, 3.8, 2.5], dtype=torch.float32)
    criterion = get_loss_function("weighted_cross_entropy", weights=weights)

    # Sample 0: wrong prediction (predicts class 2 for both batches)
    # Sample 1: correct prediction (predicts class 1 for both batches)
    logits = torch.tensor([
        [0.0, 0.0, 5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0],
    ])

    # Batch A: Sample 0 true class is 0 (majority, weight = 0.4)
    target_majority_err = torch.tensor([0, 1])
    loss_majority = criterion(logits, target_majority_err)

    # Batch B: Sample 0 true class is 3 (minority, weight = 3.8)
    target_minority_err = torch.tensor([3, 1])
    loss_minority = criterion(logits, target_minority_err)

    # Minority error weighted penalty (3.3025) exceeds majority error penalty (0.8599)
    assert loss_minority.item() > loss_majority.item()


def test_e003_data_loaders_isolation():
    """Verify E003 loaders only access Train and Validation, strictly isolating test.csv."""
    base_cfg = load_config("configs/base.yaml")
    exp_cfg = load_config("configs/experiments/class_balance.yaml")
    base_cfg["compute"]["num_workers"] = 0

    train_loader, val_loader = create_e003_data_loaders(base_cfg, exp_cfg)

    train_csv = Path("data/processed/aptos/train.csv")
    val_csv = Path("data/processed/aptos/validation.csv")
    test_csv = Path("data/processed/aptos/test.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    expected_train = len(train_df[train_df.get("is_conflicting_duplicate", False) != True])
    expected_val = len(val_df[val_df.get("is_conflicting_duplicate", False) != True])

    assert len(train_loader.dataset) == expected_train
    assert len(val_loader.dataset) == expected_val

    train_ids = set(train_loader.dataset.df["id_code"])
    val_ids = set(val_loader.dataset.df["id_code"])
    test_ids = set(test_df["id_code"])

    assert train_ids.isdisjoint(test_ids), "test.csv IDs leaked into train loader!"
    assert val_ids.isdisjoint(test_ids), "test.csv IDs leaked into validation loader!"


def test_e001_and_e002_artifacts_remain_intact():
    """Confirm E001 baseline and E002 experiment artifacts remain intact."""
    assert Path("models/checkpoints/E001_best_model.pt").is_file(), "E001 checkpoint missing"
    assert Path("experiments/E001_baseline/metrics.json").is_file(), "E001 metrics missing"
    assert Path("experiments/E002_preprocessing/README.md").is_file(), "E002 documentation missing"

    # If E002 was trained locally, ensure its checkpoint is still valid
    e002_ckpt = Path("models/checkpoints/E002_best_model.pt")
    if e002_ckpt.exists():
        assert e002_ckpt.is_file()