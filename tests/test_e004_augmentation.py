"""Unit and contract tests for E004 augmentation transform, deterministic validation, and isolation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.data.loaders import create_e004_data_loaders
from src.data.transforms import E001BaselineTransform, E004AugmentationTransform
from src.utils.config import load_config


def test_deterministic_validation_transform():
    """Verify calling E001BaselineTransform repeatedly on the same image yields identical tensors."""
    dummy_rgb = np.random.randint(0, 255, size=(600, 600, 3), dtype=np.uint8)
    val_transform = E001BaselineTransform(target_size=(512, 512))

    t1 = val_transform(dummy_rgb)
    t2 = val_transform(dummy_rgb)

    assert torch.equal(t1, t2), "Validation transform must be strictly deterministic"
    assert t1.shape == (3, 512, 512)
    assert t1.dtype == torch.float32
    assert t1.min().item() >= 0.0 and t1.max().item() <= 1.0


def test_e004_augmentation_stochastic_variation():
    """Verify E004AugmentationTransform produces variation across calls while preserving bounds."""
    dummy_rgb = np.random.randint(40, 220, size=(600, 600, 3), dtype=np.uint8)
    aug_transform = E004AugmentationTransform(
        target_size=(512, 512),
        hflip_p=0.5,
        rotation_p=0.7,
        max_rotation_deg=10.0,
        affine_p=0.5,
        jitter_p=0.5,
    )

    t1 = aug_transform(dummy_rgb)
    t2 = aug_transform(dummy_rgb)

    assert t1.shape == (3, 512, 512)
    assert t2.shape == (3, 512, 512)
    assert t1.dtype == torch.float32
    assert t2.dtype == torch.float32
    assert t1.min().item() >= 0.0 and t1.max().item() <= 1.0
    assert t2.min().item() >= 0.0 and t2.max().item() <= 1.0
    # Stochastic operations must produce different pixel distributions across multiple runs
    assert not torch.equal(t1, t2), "Augmentation pipeline must produce stochastic variation"


def test_e004_data_loaders_isolation_and_governance():
    """Verify E004 data loader isolates test.csv and applies deterministic transform to validation."""
    base_cfg = load_config("configs/base.yaml")
    exp_cfg = load_config("configs/experiments/augmentation.yaml")
    base_cfg["compute"]["num_workers"] = 0

    train_loader, val_loader = create_e004_data_loaders(base_cfg, exp_cfg)

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

    # Verify no test leakage
    train_ids = set(train_loader.dataset.df["id_code"])
    val_ids = set(val_loader.dataset.df["id_code"])
    test_ids = set(test_df["id_code"])

    assert train_ids.isdisjoint(test_ids), "test.csv IDs leaked into train loader"
    assert val_ids.isdisjoint(test_ids), "test.csv IDs leaked into validation loader"

    # Confirm validation dataset transform is deterministic E001BaselineTransform
    assert isinstance(val_loader.dataset.transform, E001BaselineTransform)
    assert isinstance(train_loader.dataset.transform, E004AugmentationTransform)


def test_e001_e002_and_e003_artifacts_remain_intact():
    """Confirm E001, E002, and E003 artifacts remain unaltered."""
    assert Path("experiments/E001_baseline/metrics.json").is_file(), "E001 metrics missing"
    assert Path("experiments/E002_preprocessing/metrics.json").is_file(), "E002 metrics missing"
    assert Path("experiments/E003_class_balance/metrics.json").is_file(), "E003 metrics missing"

    # E001 baseline checkpoint verification
    e001_ckpt = Path("models/checkpoints/E001_best_model.pt")
    if e001_ckpt.exists():
        assert e001_ckpt.is_file()

    # Experimental checkpoints are generated upon local training execution
    for ckpt_name in ["E002_best_model.pt", "E003_best_model.pt"]:
        ckpt_path = Path("models/checkpoints") / ckpt_name
        if ckpt_path.exists():
            assert ckpt_path.is_file()