from pathlib import Path
import json
import torch
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_validation_cohort_count_is_601():
    val_path = ROOT_DIR / "data/processed/aptos/validation.csv"
    assert val_path.is_file()
    df = pd.read_csv(val_path)
    assert len(df) == 601

def test_test_partition_isolation_in_eval_script():
    script_path = ROOT_DIR / "scripts/evaluate_e006_current_validation.py"
    assert script_path.is_file()
    content = script_path.read_text()
    assert "test.csv" not in content

def test_e006_checkpoint_exists_and_records_canonical_metric():
    ckpt_path = ROOT_DIR / "models/checkpoints/E006_best_model.pt"
    assert ckpt_path.is_file()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_metric = ckpt.get("best_metric", 0.0)
    assert pytest.approx(best_metric, 0.001) == 0.5970

def test_e004_checkpoint_remains_absent():
    e004_path = ROOT_DIR / "models/checkpoints/E004_best_model.pt"
    assert not e004_path.is_file(), "E004 checkpoint unexpectedly exists; must remain absent"

def test_invalid_standalone_result_is_labeled_superseded():
    readme_path = ROOT_DIR / "experiments/E006_current_validation/README.md"
    assert readme_path.is_file()
    content = readme_path.read_text()
    assert "SUPERSEDED" in content or "INVALID" in content
    assert "0.5970" in content

def test_canonical_e006_metrics_in_experiment_readme():
    readme_path = ROOT_DIR / "experiments/E006_ordinal/README.md"
    assert readme_path.is_file()
    content = readme_path.read_text()
    assert "0.5970" in content
    assert "REJECTED" in content
