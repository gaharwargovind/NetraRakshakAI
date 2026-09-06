import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassFocalLoss(nn.Module):
    """
    Numerically stable Multiclass Focal Loss without class weighting.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Default: 2.0.
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted unnormalized scores of shape (N, C).
            targets: Ground-truth class labels of shape (N,) with integer indices in [0, C-1].
        """
        # Log-softmax for numerical stability
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        # Gather target probabilities
        targets_expanded = targets.view(-1, 1)
        log_pt = log_p.gather(dim=-1, index=targets_expanded).squeeze(-1)
        pt = p.gather(dim=-1, index=targets_expanded).squeeze(-1)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
