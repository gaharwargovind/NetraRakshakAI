import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            else:
                alpha_tensor = alpha.clone().detach().float()
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        targets_expanded = targets.view(-1, 1)
        log_pt = log_p.gather(dim=-1, index=targets_expanded).squeeze(-1)
        pt = p.gather(dim=-1, index=targets_expanded).squeeze(-1)

        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha_device = self.alpha.to(logits.device)
            alpha_t = alpha_device.gather(dim=0, index=targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
