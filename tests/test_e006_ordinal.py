"""Unit and contract tests for E006 Cumulative Ordinal Regression."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.loaders import create_e006_data_loaders
from src.evaluation.metrics import (
    compute_ordinal_error_metrics,
    decode_ordinal_probabilities,
    reconstruct_ordinal_class_probabilities,
)
from src.models.ordinal import DROrdinalNet, OrderedThresholdHead
from src.training.losses import (
    CumulativeOrdinalBCEWithLogitsLoss,
    compute_ordinal_task_weights,
    encode_ordinal_targets,
)
from src.utils.config import load_config


def test_ordered_threshold_head_monotonicity_guarantee():
    """Verify that z_1 > z_2 > z_3 > z_4 and p_1 > p_2 > p_3 > p_4 hold unconditionally."""
    torch.manual_seed(42)
    head = OrderedThresholdHead(in_features=1280, num_tasks=4, dropout_rate=0.0)

    # Test over 20 random batches of features
    for _ in range(20):
        dummy_feat = torch.randn(32, 1280) * 5.0
        logits = head(dummy_feat)
        probs = torch.sigmoid(logits)

        # Verify strictly decreasing logits: z_k > z_{k+1}
        diffs_logits = logits[:, :-1] - logits[:, 1:]
        assert torch.all(diffs_logits > 0), "Logits failed monotonicity invariant"

        # Verify strictly decreasing cumulative probabilities: p_k > p_{k+1}
        diffs_probs = probs[:, :-1] - probs[:, 1:]
        assert torch.all(diffs_probs > 0), "Probabilities failed monotonicity invariant"


def test_natural_probability_reconstruction_and_unit_sum():
    """
    Test reconstruction using explicitly ordered cumulative probabilities:
        p = [0.90, 0.70, 0.30, 0.10]
        Expected: P0=0.10, P1=0.20, P2=0.40, P3=0.20, P4=0.10
    Verify all P >= 0, sum == 1.0 within tolerance, and mathematically correct.
    """
    p = torch.tensor([[0.90, 0.70, 0.30, 0.10]], dtype=torch.float32)
    p_class = reconstruct_ordinal_class_probabilities(p)

    expected = torch.tensor([[0.10, 0.20, 0.40, 0.20, 0.10]], dtype=torch.float32)

    assert p_class.shape == (1, 5)
    assert torch.all(p_class >= 0.0), "All reconstructed probabilities must be non-negative"
    assert torch.allclose(p_class.sum(dim=-1), torch.tensor([1.0]), atol=1e-6), "Probabilities must sum to 1.0"
    assert torch.allclose(p_class, expected, atol=1e-5), f"Reconstruction mismatch: got {p_class}, expected {expected}"


def test_ordinal_decoding_exact_test_vectors():
    """
    Test discrete grade decoding with fixed tau=0.5 against specified test vectors:
        [0.2, 0.1, 0.05, 0.01] -> grade 0
        [0.8, 0.2, 0.1, 0.01]  -> grade 1
        [0.9, 0.8, 0.2, 0.1]   -> grade 2
        [0.9, 0.8, 0.7, 0.2]   -> grade 3
        [0.9, 0.8, 0.7, 0.6]   -> grade 4
    """
    probs = torch.tensor([
        [0.2, 0.1, 0.05, 0.01],
        [0.8, 0.2, 0.1, 0.01],
        [0.9, 0.8, 0.2, 0.1],
        [0.9, 0.8, 0.7, 0.2],
        [0.9, 0.8, 0.7, 0.6],
    ], dtype=torch.float32)

    grades = decode_ordinal_probabilities(probs, threshold=0.5)
    expected_grades = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    assert torch.equal(grades, expected_grades), f"Decoded grades {grades} did not match {expected_grades}"


def test_ordinal_target_encoding():
    """Verify conversion of labels [0, 1, 2, 3, 4] to cumulative binary targets."""
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    targets = encode_ordinal_targets(labels, num_classes=5)

    expected = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float32)

    assert targets.shape == (5, 4)
    assert torch.equal(targets, expected)


def test_cumulative_ordinal_loss_and_gradient_flow():
    """Verify CumulativeOrdinalBCEWithLogitsLoss produces scalar finite loss and backpropagates gradients."""
    pos_weights = torch.tensor([0.9712, 1.4610, 6.4906, 11.3557], dtype=torch.float32)
    criterion = CumulativeOrdinalBCEWithLogitsLoss(pos_weights=pos_weights)

    head = OrderedThresholdHead(in_features=128, num_tasks=4)
    feat = torch.randn(8, 128, requires_grad=True)
    targets = encode_ordinal_targets(torch.tensor([0, 1, 2, 3, 4, 0, 2, 4]), num_classes=5)

    logits = head(feat)
    loss = criterion(logits, targets)

    assert torch.isfinite(loss)
    assert loss.dim() == 0

    loss.backward()
    assert head.linear.weight.grad is not None
    assert head.initial_cutoff.grad is not None
    assert head.log_cutoffs.grad is not None


def test_e006_validation_cohort_fail_fast():
    """Verify validation loader enforces exactly 601 active validation samples."""
    base_cfg = load_config("configs/base.yaml")
    exp_cfg = load_config("configs/experiments/ordinal.yaml")
    base_cfg["compute"]["num_workers"] = 0

    train_loader, val_loader = create_e006_data_loaders(base_cfg, exp_cfg)

    # Active validation cohort assertion
    assert len(val_loader.dataset) == 601, f"Active validation count must be 601, got {len(val_loader.dataset)}"
    assert len(train_loader.dataset) == 2397, f"Train count must be 2397, got {len(train_loader.dataset)}"


def test_existing_artifacts_sha256_unaltered():
    """Verify SHA-256 hashes of E001-E005 baseline metrics match the pre-implementation audit."""
    import hashlib

    expected_hashes = {
        "experiments/E001_baseline/metrics.json": "3731c658d0839b20f90a2fa27d5bd1be3814e69989fdf3f71adf80d9f1771e23",
        "experiments/E002_preprocessing/metrics.json": "82e2092f27af4edad9449e802d1cfca02379ddf90df853b1e18d95a84e6aab63",
        "experiments/E003_class_balance/metrics.json": "fdf41e0e4c3a496fa6f0ccc6c639343e4adcfb73fcc6206ba9da16157fdda91b",
        "experiments/E004_augmentation/metrics.json": "a4aa25b13cfdef20aa32e7f8be180b61c6429ad3b7ddc2b11e3d8dd5f90c7e9a",
        "experiments/E005_architecture/metrics.json": "944c7cb45c74063dad28364c263bcb4e9273b53d988d48c2d5c2feefd043ceac",
    }

    for path_str, exp_hash in expected_hashes.items():
        path = Path(path_str)
        assert path.is_file(), f"Artifact missing: {path}"
        with open(path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == exp_hash, f"Integrity violation on {path_str}!"