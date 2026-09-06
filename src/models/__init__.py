"""Model architectures and unified backbone instantiation factory."""

from typing import Optional
import torch.nn as nn

from src.models.convnext import DRConvNeXt
from src.models.efficientnet import DREfficientNet
from src.models.ordinal import DROrdinalNet, OrderedThresholdHead
from src.models.quality_model import DRQualityModel
from src.models.resnet import DRResNet

__all__ = [
    "DREfficientNet",
    "DRResNet",
    "DRConvNeXt",
    "DROrdinalNet",
    "OrderedThresholdHead",
    "DRQualityModel",
    "get_model",
]


def get_model(
    backbone_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    head_type: str = "classification",
) -> nn.Module:
    """
    Model factory to instantiate candidate backbones and head architectures.

    Args:
        backbone_name: Backbone architecture identifier
            ('efficientnet_b0', 'resnet50', 'convnext_tiny', 'ordinal_efficientnet_b0').
        num_classes: Number of target ICDR severity classes (default: 5).
        pretrained: If True, loads official Torchvision ImageNet weights.
        dropout_rate: Dropout probability preceding the classification/ordinal projection.
        head_type: Head architecture variant ('classification' or 'ordered_threshold').

    Returns:
        Configured PyTorch nn.Module instance.

    Raises:
        ValueError: If backbone_name or head_type is unsupported.
    """
    name = backbone_name.lower().strip()
    head = head_type.lower().strip()

    # Route ordinal cumulative regression head
    if head == "ordered_threshold" or name in ["ordinal_efficientnet_b0", "ordinal"]:
        return DROrdinalNet(
            backbone_name="efficientnet_b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )

    # Standard categorical classification backbones (E001-E005)
    if name == "efficientnet_b0":
        return DREfficientNet(
            backbone_name="efficientnet_b0",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif name in ["resnet50", "resnet_50"]:
        return DRResNet(
            layers=50,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )
    elif name in ["convnext_tiny", "convnext"]:
        return DRConvNeXt(
            variant="convnext_tiny",
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )

    raise ValueError(
        f"Unsupported backbone_name '{backbone_name}' with head_type '{head_type}'. "
        f"Supported backbones: ['efficientnet_b0', 'resnet50', 'convnext_tiny', 'ordinal_efficientnet_b0']. "
        f"Supported heads: ['classification', 'ordered_threshold']."
    )