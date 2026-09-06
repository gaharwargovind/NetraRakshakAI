"""ConvNeXt-Tiny architecture wrapper for 5-class ICDR diabetic retinopathy grading."""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models


class DRConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny backbone fine-tuning wrapper.
    Replaces classifier linear projection with Dropout(p=0.2) + Linear(768, 5).
    """

    def __init__(
        self,
        variant: str = "convnext_tiny",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate

        if variant != "convnext_tiny":
            raise ValueError(f"Only convnext_tiny is supported in E005, got variant={variant}")

        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = tv_models.convnext_tiny(weights=weights)

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=in_features, out_features=self.num_classes, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class logits of shape [batch_size, 5]."""
        return self.model(x)