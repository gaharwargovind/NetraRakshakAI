"""Unit tests for configuration parsing, schema validation, and path handling."""

import pytest
from pathlib import Path
from src.utils.config import load_config, validate_config_structure


def test_base_config_loading():
    """Verify that base.yaml exists, loads cleanly, and contains required schema."""
    config_path = Path("configs/base.yaml")
    assert config_path.is_file(), "configs/base.yaml must exist on disk"
    
    cfg = load_config(config_path)
    assert isinstance(cfg, dict)
    
    required_keys = ["project", "compute", "data", "quality_gate", "uncertainty", "targets"]
    assert validate_config_structure(cfg, required_keys) is True
    
    assert cfg["project"]["sih_problem_id"] == "SIH26038"
    assert cfg["data"]["num_classes"] == 5
    assert cfg["data"]["referable_threshold_grade"] == 2


def test_missing_config_raises():
    """Verify that a missing config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_file.yaml")


def test_experiment_configs_exist():
    """Verify that all scheduled experiment YAML files exist."""
    exp_dir = Path("configs/experiments")
    assert exp_dir.is_dir()
    
    expected_configs = [
        "baseline.yaml",
        "preprocessing.yaml",
        "class_balance.yaml",
        "augmentation.yaml",
        "architecture.yaml",
        "ordinal.yaml",
        "calibration.yaml",
        "robustness.yaml",
    ]
    for filename in expected_configs:
        path = exp_dir / filename
        assert path.is_file(), f"Missing expected experiment config: {filename}"
        cfg = load_config(path)
        assert "experiment" in cfg