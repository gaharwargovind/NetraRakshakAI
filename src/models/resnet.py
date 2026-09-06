"""ResNet-50 architecture wrapper for 5-class ICDR diabetic retinopathy grading."""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models


class DRResNet(nn.Module):
    """
    ResNet-50 backbone fine-tuning wrapper.
    Replaces default 1000-class fc head with Dropout(p=0.2) + Linear(2048, 5).
    """

    def __init__(
        self,
        layers: int = 50,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        if layers != 50:
            raise ValueError(f"Only ResNet-50 is supported in E005, got layers={layers}")

        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = tv_models.resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape [batch_size, 5]."""
        return self.model(x)