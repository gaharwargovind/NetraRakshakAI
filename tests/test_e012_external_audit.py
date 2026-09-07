import hashlib
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

def test_e007_checkpoint_hash_unmodified():
    ckpt_path = ROOT_DIR / "models/checkpoints/E007_best_model.pt"
    assert ckpt_path.is_file()
    expected_sha = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"
    current_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert current_sha == expected_sha, "E007 checkpoint hash changed"

def test_aptos_partitions_intact():
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    test_csv = ROOT_DIR / "data/processed/aptos/test.csv"
    assert len(pd.read_csv(val_csv)) == 601
    test_lines = sum(1 for _ in open(test_csv)) - 1
    assert test_lines == 602

def test_protocol_and_dataset_card_exist():
    card = ROOT_DIR / "docs/external_dataset_card.md"
    proto = ROOT_DIR / "docs/external_validation_protocol.md"
    assert card.is_file()
    assert proto.is_file()

def test_label_mapping_integrity():
    map_path = ROOT_DIR / "experiments/E012_external_validation/label_mapping.csv"
    if map_path.is_file():
        df = pd.read_csv(map_path)
        assert len(df) == 5
        assert set(df["mapped_icdr_grade"]) == {0, 1, 2, 3, 4}
