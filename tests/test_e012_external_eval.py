import json, hashlib
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_predictions_csv_structure():
    p = ROOT_DIR / "experiments/E012_external_validation/external_predictions.csv"
    if p.is_file():
        df = pd.read_csv(p)
        assert len(df) == 455
        assert "predicted_grade" in df.columns
        assert "true_grade" in df.columns
        assert set(df["predicted_grade"]).issubset({0, 1, 2, 3, 4})

def test_e007_checkpoint_invariant():
    p = ROOT_DIR / "models/checkpoints/E007_best_model.pt"
    expected = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected
