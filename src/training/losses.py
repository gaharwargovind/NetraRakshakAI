"""Loss function factories and training-derived class weight computation."""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Union[List[int], np.ndarray, pd.Series, torch.Tensor],
    num_classes: int = 5,
) -> torch.Tensor:
    """
    Compute balanced inverse-frequency class weights strictly from training labels:
        w_c = N / (K * N_c)
    where:
        N = total training samples
        K = number of classes
        N_c = count of training samples in class c

    Args:
        labels: 1D collection of integer class labels in [0, num_classes - 1].
        num_classes: Number of distinct target classes (default: 5).

    Returns:
        torch.Tensor of shape (num_classes,) with dtype float32.

    Raises:
        ValueError: If labels collection is empty, contains invalid values,
                    or any class has 0 occurrences.
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    elif isinstance(labels, (pd.Series, list)):
        labels_np = np.asarray(labels)
    elif isinstance(labels, np.ndarray):
        labels_np = labels
    else:
        raise TypeError(f"Unsupported type for labels: {type(labels)}")

    if labels_np.ndim != 1 or len(labels_np) == 0:
        raise ValueError(f"Labels must be a non-empty 1D array, got shape {labels_np.shape}")

    if not np.issubdtype(labels_np.dtype, np.integer):
        labels_np = labels_np.astype(int)

    if np.any(labels_np < 0) or np.any(labels_np >= num_classes):
        raise ValueError(f"Labels must be within [0, {num_classes - 1}]")

    n_total = len(labels_np)
    counts = np.bincount(labels_np, minlength=num_classes)

    zero_classes = np.where(counts == 0)[0]
    if len(zero_classes) > 0:
        raise ValueError(f"Classes with 0 samples detected: {zero_classes.tolist()}. Cannot compute inverse weights.")

    weights = n_total / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def get_loss_function(
    loss_type: str = "cross_entropy",
    weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Factory creating configured loss criterion.

    Args:
        loss_type: Loss identifier ("cross_entropy", "weighted_cross_entropy").
        weights: Optional tensor of class weights of shape (num_classes,).

    Returns:
        torch.nn.Module loss instance.
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=None)
    elif loss_type == "weighted_cross_entropy":
        if weights is None:
            raise ValueError("weights tensor must be provided for 'weighted_cross_entropy'")
        return nn.CrossEntropyLoss(weight=weights)
    raise ValueError(f"Unsupported loss_type '{loss_type}'. Supported: ['cross_entropy', 'weighted_cross_entropy']")

def encode_ordinal_targets(labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Encode integer label y into binary cumulative targets Y_k = I(y >= k) for k in [1, num_classes - 1]."""
    num_tasks = num_classes - 1
    # tasks: 1, 2, 3, 4
    cutoffs = torch.arange(1, num_classes, device=labels.device).unsqueeze(0)  # [1, 4]
    targets = (labels.unsqueeze(1) >= cutoffs).to(torch.float32)  # [batch_size, 4]
    return targets


def compute_ordinal_task_weights(labels: Union[np.ndarray, pd.Series, torch.Tensor], num_classes: int = 5) -> torch.Tensor:
    """Compute pos_weight_k = (N - N_k^+) / N_k^+ strictly from training labels."""
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    n_total = len(labels_np)
    pos_weights = []
    for k in range(1, num_classes):
        pos_count = np.sum(labels_np >= k)
        neg_count = n_total - pos_count
        pos_weights.append(neg_count / float(pos_count))

    return torch.tensor(pos_weights, dtype=torch.float32)


class CumulativeOrdinalBCEWithLogitsLoss(nn.Module):
    """Multi-task binary cross-entropy loss with task-specific positive weights for cumulative tasks."""

    def __init__(self, pos_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.register_buffer("pos_weights", pos_weights if pos_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weights,
            reduction="mean",
        )
        return bce