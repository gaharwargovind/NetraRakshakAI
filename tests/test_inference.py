"""Contract tests for quality gate and screening inference interfaces."""

import pytest
import numpy as np
from src.inference.quality_gate import QualityGateDecision
from src.inference.predictor import ScreeningPredictor


def test_quality_gate_contract():
    """Verify QualityGateDecision constructor and method contract."""
    gate = QualityGateDecision(max_recapture_attempts=3)
    assert gate.max_attempts == 3
    
    dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        gate.check(dummy_image, attempt_count=1)


def test_predictor_contract():
    """Verify ScreeningPredictor initialization and method contract."""
    predictor = ScreeningPredictor(config={"device": "cpu"})
    assert predictor.config["device"] == "cpu"
    
    dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        predictor.predict(dummy_image)