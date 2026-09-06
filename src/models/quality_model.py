"""Neural Image Quality Assessment auxiliary model interface."""
import torch
import torch.nn as nn


class DRQualityModel(nn.Module):
    """Auxiliary neural image gradability assessment model."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Neural quality assessment is scheduled for Phase 8.")