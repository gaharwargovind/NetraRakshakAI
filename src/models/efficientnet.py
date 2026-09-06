"""EfficientNet-B0 backbone architecture for 5-class ICDR severity classification."""

import logging
import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)


class DREfficientNet(nn.Module):
    """
    Torchvision EfficientNet-B0 fine-tuning backbone.
    Replaces 1000-class head with dropout(0.2) + 5-class linear projection.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        if backbone_name != "efficientnet_b0":
            raise ValueError(f"DREfficientNet for E001 requires 'efficientnet_b0', got: {backbone_name}")

        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = tv_models.efficientnet_b0(weights=weights)

        # Standard torchvision EfficientNet classifier is Sequential(Dropout, Linear)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw uncalibrated class logits of shape [batch_size, 5]."""
        return self.model(x)