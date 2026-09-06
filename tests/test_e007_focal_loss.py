from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

from src.losses.focal_loss import MulticlassFocalLoss

def test_focal_loss_gamma_zero_matches_cross_entropy():
    """When gamma=0, MulticlassFocalLoss must match CrossEntropyLoss exactly."""
    fl = MulticlassFocalLoss(gamma=0.0, reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")

    logits = torch.randn(10, 5, requires_grad=True)
    targets = torch.randint(0, 5, (10,))

    loss_fl = fl(logits, targets)
    loss_ce = ce(logits, targets)

    assert torch.allclose(loss_fl, loss_ce, atol=1e-6)

def test_focal_loss_downweights_easy_examples():
    """Focal loss with gamma=2.0 must produce strictly lower loss on high-confidence correct predictions."""
    fl = MulticlassFocalLoss(gamma=2.0, reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")

    # High confidence correct: target class has large positive logit
    logits = torch.tensor([[10.0, -2.0, -2.0, -2.0, -2.0]])
    targets = torch.tensor([0])

    loss_fl = fl(logits, targets)
    loss_ce = ce(logits, targets)

    assert loss_fl.item() < loss_ce.item()
    # At p_t ≈ 0.9999, (1 - p_t)^2 should downweight the loss by orders of magnitude
    assert loss_fl.item() < 1e-4

def test_focal_loss_numerical_stability():
    """Check stability across extreme dynamic ranges."""
    fl = MulticlassFocalLoss(gamma=2.0, reduction="mean")
    logits = torch.tensor([[-100.0, 100.0, 0.0, 50.0, -50.0]], dtype=torch.float32)
    targets = torch.tensor([1])

    loss = fl(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_e007_validation_manifest_count():
    val_path = ROOT_DIR / "data/processed/aptos/validation.csv"
    assert val_path.is_file()
    df = pd.read_csv(val_path)
    assert len(df) == 601

def test_e007_test_set_isolation():
    script_path = ROOT_DIR / "scripts/train_e007.py"
    if script_path.is_file():
        content = script_path.read_text()
        assert "data/processed/aptos/test.csv" not in content
        assert "test.csv" not in content

def test_e004_artifacts_preserved():
    e004_metrics = ROOT_DIR / "experiments/E004_augmentation/metrics.json"
    assert e004_metrics.is_file()
    assert e004_metrics.stat().st_size > 0
