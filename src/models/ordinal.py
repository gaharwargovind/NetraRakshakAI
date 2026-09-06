"""Ordered-threshold cumulative ordinal regression head and model wrapper for DR grading."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class OrderedThresholdHead(nn.Module):
    """
    Cumulative ordinal regression head with guaranteed monotonic cutoffs:
        s(x) = w^T x
        c_1 = beta_1
        c_k = c_{k-1} + softplus(delta_k)  for k in {2, 3, 4}
        z_k = s(x) - c_k

    Because softplus(delta) > 0 unconditionally:
        c_1 < c_2 < c_3 < c_4
        z_1 > z_2 > z_3 > z_4  for every input x
        P(y >= 1) > P(y >= 2) > P(y >= 3) > P(y >= 4)
    """

    def __init__(self, in_features: int = 1280, num_tasks: int = 4, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_tasks = num_tasks

        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear = nn.Linear(in_features, 1, bias=False)

        # Unconstrained learnable parameters for cutoffs
        self.initial_cutoff = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_cutoffs = nn.Parameter(torch.zeros(num_tasks - 1, dtype=torch.float32))

    def get_cutoffs(self) -> torch.Tensor:
        """Compute ordered cutoffs: [c_1, c_2, c_3, c_4]."""
        deltas = F.softplus(self.log_cutoffs)
        c1 = self.initial_cutoff.unsqueeze(0)
        c_remaining = c1 + torch.cumsum(deltas, dim=0)
        return torch.cat([c1, c_remaining], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature tensor of shape [batch_size, in_features].
        Returns:
            Cumulative logits of shape [batch_size, 4] where z_1 > z_2 > z_3 > z_4.
        """
        x = self.dropout(x)
        latent_score = self.linear(x)  # [batch_size, 1]
        cutoffs = self.get_cutoffs().unsqueeze(0)  # [1, num_tasks]
        logits = latent_score - cutoffs  # [batch_size, num_tasks]
        return logits


class DROrdinalNet(nn.Module):
    """EfficientNet-B0 backbone coupled with guaranteed monotonic OrderedThresholdHead."""

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone_name != "efficientnet_b0":
            raise ValueError(f"E006 uses EfficientNet-B0 control backbone, got {backbone_name}")
        if num_classes != 5:
            raise ValueError(f"Expected 5 ICDR classes, got {num_classes}")

        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = tv_models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        self.ordinal_head = OrderedThresholdHead(
            in_features=in_features,
            num_tasks=num_classes - 1,  # 4 cumulative binary tasks
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return cumulative logits of shape [batch_size, 4]."""
        features = self.backbone(x)
        return self.ordinal_head(features)