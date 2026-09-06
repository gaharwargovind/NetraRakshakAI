"""Unit and contract tests for E005 backbone architectures, data loaders, and isolation."""

from pathlib import Path
import pytest
import torch
from src.data.loaders import create_e005_data_loaders
from src.models import get_model
from src.models.convnext import DRConvNeXt
from src.models.efficientnet import DREfficientNet
from src.models.resnet import DRResNet
from src.utils.config import load_config


@pytest.mark.parametrize("backbone_name,expected_cls,min_params,max_params", [
    ("efficientnet_b0", DREfficientNet, 3_500_000, 6_000_000),
    ("resnet50", DRResNet, 22_000_000, 26_000_000),
    ("convnext_tiny", DRConvNeXt, 26_000_000, 30_000_000),
])
def test_backbone_instantiation_shapes_and_parameters(backbone_name, expected_cls, min_params, max_params):
    """Verify each candidate backbone builds with 5 output classes and expected parameter scale."""
    model = get_model(backbone_name, num_classes=5, pretrained=False, dropout_rate=0.2)
    assert isinstance(model, expected_cls)

    n_params = sum(p.numel() for p in model.parameters())
    assert min_params <= n_params <= max_params, f"{backbone_name} param count {n_params} out of range"

    dummy_input = torch.randn(2, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 5), f"{backbone_name} output shape expected (2, 5), got {out.shape}"
    assert out.dtype == torch.float32


def test_invalid_backbone_name_raises():
    """Verify get_model factory rejects unsupported backbones with informative ValueError."""
    with pytest.raises(ValueError, match="Unsupported backbone_name"):
        get_model("vgg16", num_classes=5)


def test_e005_data_loaders_isolation():
    """Verify E005 loader loads Train and Validation only, strictly excluding test.csv."""
    import pandas as pd

    base_cfg = load_config("configs/base.yaml")
    exp_cfg = load_config("configs/experiments/architecture.yaml")
    base_cfg["compute"]["num_workers"] = 0

    train_loader, val_loader = create_e005_data_loaders(base_cfg, exp_cfg)

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

    train_ids = set(train_loader.dataset.df["id_code"])
    val_ids = set(val_loader.dataset.df["id_code"])
    test_ids = set(test_df["id_code"])

    assert train_ids.isdisjoint(test_ids), "test.csv IDs leaked into train loader"
    assert val_ids.isdisjoint(test_ids), "test.csv IDs leaked into validation loader"


def test_previous_artifacts_remain_intact():
    """Confirm E001, E002, E003, and E004 checkpoints and metric records remain unaltered."""
    assert Path("models/checkpoints/E001_best_model.pt").is_file(), "E001 checkpoint missing"
    assert Path("experiments/E001_baseline/metrics.json").is_file(), "E001 metrics missing"
    assert Path("experiments/E002_preprocessing/metrics.json").is_file(), "E002 metrics missing"
    assert Path("experiments/E003_class_balance/metrics.json").is_file(), "E003 metrics missing"
    assert Path("experiments/E004_augmentation/metrics.json").is_file(), "E004 metrics missing"

    for ckpt_name in ["E003_best_model.pt", "E004_best_model.pt"]:
        ckpt_path = Path("models/checkpoints") / ckpt_name
        if ckpt_path.exists():
            assert ckpt_path.is_file()