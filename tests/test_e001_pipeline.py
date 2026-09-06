"""Comprehensive contract and unit tests for E001 baseline components."""

from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.transforms import E001BaselineTransform
from src.evaluation.metrics import compute_clinical_screening_metrics
from src.models.efficientnet import DREfficientNet
from src.training.early_stopping import EarlyStopping
from src.training.losses import get_loss_function
from src.training.scheduler import get_lr_scheduler


def test_efficientnet_output_shape():
    """Verify DREfficientNet accepts standard tensor and outputs [B, 5] logits."""
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=False, dropout_rate=0.2)
    model.eval()
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (2, 5), f"Expected shape (2, 5), got {out.shape}"


def test_baseline_transforms_shape_and_range():
    """Verify E001 baseline transform outputs shape (3, 512, 512) within [0, 1]."""
    dummy_image = np.full((600, 800, 3), fill_value=128, dtype=np.uint8)
    transform = E001BaselineTransform(target_size=(512, 512))
    tensor = transform(dummy_image)

    assert tensor.shape == (3, 512, 512)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_loss_factory():
    """Verify loss factory correctly returns standard CrossEntropyLoss."""
    criterion = get_loss_function("cross_entropy")
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    with pytest.raises(ValueError):
        get_loss_function("unknown_loss")


def test_scheduler_factory():
    """Verify scheduler factory creates CosineAnnealingLR."""
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_lr_scheduler(optimizer, {"training": {"lr_scheduler": "cosine_annealing"}}, total_epochs=10)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_early_stopping_behavior():
    """Verify EarlyStopping honors patience and detects metric improvement."""
    stopper = EarlyStopping(patience=3, mode="max")
    # Score improvements
    stop, is_best = stopper.step(0.50)
    assert not stop and is_best
    stop, is_best = stopper.step(0.55)
    assert not stop and is_best

    # Score plateaus
    stop, is_best = stopper.step(0.54)
    assert not stop and not is_best
    stop, is_best = stopper.step(0.53)
    assert not stop and not is_best
    stop, is_best = stopper.step(0.52)
    assert stop and not is_best, "Early stopping should trigger on 3rd non-improving step"


def test_clinical_metrics_computation():
    """Verify accuracy, macro F1, QWK, and referable sensitivity/specificity."""
    # Synthetic ground truth: classes 0, 1, 2, 3, 4
    y_true = np.array([0, 1, 2, 3, 4])
    # Predicted probabilities matching true labels exactly
    y_probs = np.eye(5)

    metrics = compute_clinical_screening_metrics(y_true, y_probs, referable_threshold=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["quadratic_weighted_kappa"] == 1.0
    assert metrics["referable_sensitivity"] == 1.0
    assert metrics["referable_specificity"] == 1.0