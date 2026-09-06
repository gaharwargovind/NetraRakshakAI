"""Learning rate scheduler construction."""

from typing import Any, Dict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: Dict[str, Any],
    total_epochs: int,
) -> _LRScheduler:
    """
    Construct CosineAnnealingLR scheduler matching E001 configuration.
    """
    scheduler_type = config.get("training", {}).get("lr_scheduler", "cosine_annealing")
    if scheduler_type == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    raise ValueError(f"Unsupported scheduler '{scheduler_type}' in E001.")