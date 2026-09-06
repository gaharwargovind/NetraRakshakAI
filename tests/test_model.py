"""Contract tests for model backbone interfaces."""

import pytest
from src.models.efficientnet import DREfficientNet
from src.models.resnet import DRResNet
from src.models.convnext import DRConvNeXt
from src.models.ordinal import DROrdinalNet


def test_efficientnet_interface():
    """Verify DREfficientNet constructor contract and parameter retention."""
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=True)
    assert model.num_classes == 5
    assert model.backbone_name == "efficientnet_b0"
    assert model.pretrained is True


def test_resnet_interface():
    """Verify DRResNet constructor contract."""
    model = DRResNet(layers=50, num_classes=5)
    assert model.num_classes == 5
    assert model.layers == 50


def test_convnext_interface():
    """Verify DRConvNeXt constructor contract."""
    model = DRConvNeXt(variant="convnext_tiny", num_classes=5)
    assert model.num_classes == 5
    assert model.variant == "convnext_tiny"