import pytest
import torch
import numpy as np
from pathlib import Path
from src.evaluation.calibration import (
    TemperatureScaler,
    compute_multiclass_ece,
    compute_multiclass_metrics,
    compute_referable_calibration
)

def test_temperature_scaler_positive():
    scaler = TemperatureScaler(initial_temperature=1.5)
    assert scaler.temperature.item() > 0.0

def test_probabilities_sum_to_one():
    scaler = TemperatureScaler(initial_temperature=2.0)
    logits = torch.randn(25, 5)
    probs = scaler.predict_proba(logits)
    sums = probs.sum(dim=-1).numpy()
    assert np.allclose(sums, 1.0, atol=1e-6)

def test_argmax_invariance():
    """Temperature scaling by any T > 0 must preserve exact argmax class assignments."""
    scaler = TemperatureScaler(initial_temperature=2.7)
    logits = torch.randn(100, 5)
    uncal_preds = torch.argmax(logits, dim=-1)
    cal_probs = scaler.predict_proba(logits)
    cal_preds = torch.argmax(cal_probs, dim=-1)
    assert torch.equal(uncal_preds, cal_preds)

def test_temperature_fitting_reduces_nll():
    """Fitting on overconfident logits must reduce cross-entropy loss."""
    torch.manual_seed(42)
    # Overconfident synthetic logits
    logits = torch.randn(50, 5) * 5.0
    targets = torch.randint(0, 5, (50,))
    
    scaler = TemperatureScaler(initial_temperature=1.0)
    ce_loss = torch.nn.CrossEntropyLoss()
    initial_loss = ce_loss(logits, targets).item()
    
    opt_t = scaler.fit(logits, targets)
    final_loss = ce_loss(logits / opt_t, targets).item()
    
    assert opt_t > 0.0
    assert final_loss <= initial_loss

def test_no_model_weights_modified():
    """Ensure temperature scaler contains only the scalar parameter and does not modify backbone."""
    scaler = TemperatureScaler()
    params = list(scaler.parameters())
    assert len(params) == 1
    assert params[0].numel() == 1
