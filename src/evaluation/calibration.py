from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_multiclass_ece(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    for top-1 multiclass predictions.
    """
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == targets).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)
            
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            diff = abs(bin_acc - bin_conf)
            weight = bin_count / len(targets)
            ece += weight * diff
            mce = max(mce, diff)
            bin_data.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(diff, 4)
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "error": 0.0
            })

    return round(float(ece), 4), round(float(mce), 4), {"bins": bin_data}

def compute_referable_calibration(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Evaluates calibration for binary referable risk:
    P(referable) = P(G2) + P(G3) + P(G4), Ground Truth = (target >= 2).
    Returns: (ece, brier, nll, bin_details)
    """
    p_ref = np.sum(probs[:, 2:], axis=1)
    y_ref = (targets >= 2).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (p_ref >= bin_lower) & (p_ref <= bin_upper)
        else:
            in_bin = (p_ref >= bin_lower) & (p_ref < bin_upper)
            
        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            obs_prob = float(np.mean(y_ref[in_bin]))
            pred_prob = float(np.mean(p_ref[in_bin]))
            diff = abs(obs_prob - pred_prob)
            ece += (bin_count / len(targets)) * diff
            bin_data.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": bin_count,
                "observed_rate": round(obs_prob, 4),
                "predicted_prob": round(pred_prob, 4),
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": 0,
                "observed_rate": 0.0,
                "predicted_prob": 0.0,
            })

    brier = float(np.mean((p_ref - y_ref) ** 2))
    eps = 1e-12
    p_ref_clipped = np.clip(p_ref, eps, 1.0 - eps)
    nll = float(-np.mean(y_ref * np.log(p_ref_clipped) + (1.0 - y_ref) * np.log(1.0 - p_ref_clipped)))

    return round(float(ece), 4), round(brier, 4), round(nll, 4), {"bins": bin_data}

def compute_multiclass_metrics(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Computes multiclass Brier score and Negative Log Likelihood (NLL)."""
    N, C = probs.shape
    one_hot = np.zeros((N, C), dtype=float)
    one_hot[np.arange(N), targets] = 1.0
    
    # Multiclass Brier score: mean squared difference across all classes
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    
    eps = 1e-12
    p_true = probs[np.arange(N), targets]
    nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    return round(brier, 4), round(nll, 4)

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling (Guo et al., 2017).
    p_i = softmax(z_i / T) where T > 0 is a single scalar.
    Strictly preserves class ranking and argmax predictions.
    """
    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
        """Fits optimal scalar temperature T using L-BFGS to minimize Cross-Entropy (NLL)."""
        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            with torch.no_grad():
                self.temperature.clamp_(min=1e-3)
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
        return float(self.temperature.item())

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scaled = self.forward(logits)
            return F.softmax(scaled, dim=-1)

def plot_multiclass_reliability(
    uncal_data: Dict[str, Any],
    cal_data: Dict[str, Any],
    out_path: str,
    uncal_ece: float,
    cal_ece: float
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, ece in zip(axes, [uncal_data, cal_data], ["Uncalibrated (E007 Baseline)", "Temperature Scaled"], [uncal_ece, cal_ece]):
        bins = data["bins"]
        confs = [b["confidence"] if b["count"] > 0 else (b["range"][0] + b["range"][1]) / 2 for b in bins]
        accs = [b["accuracy"] for b in bins]
        counts = [b["count"] for b in bins]
        
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        bars = ax.bar(confs, accs, width=0.08, alpha=0.6, color="royalblue", edgecolor="black", label="Outputs")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel("Mean Predicted Confidence", fontsize=10)
        ax.set_ylabel("Empirical Accuracy", fontsize=10)
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_referable_reliability(
    uncal_data: Dict[str, Any],
    cal_data: Dict[str, Any],
    out_path: str,
    uncal_ece: float,
    cal_ece: float
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, ece in zip(axes, [uncal_data, cal_data], ["Referable Risk (Pre-Calibration)", "Referable Risk (Post-Calibration)"], [uncal_ece, cal_ece]):
        bins = data["bins"]
        preds = [b["predicted_prob"] if b["count"] > 0 else (b["range"][0] + b["range"][1]) / 2 for b in bins]
        obs = [b["observed_rate"] for b in bins]
        
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
        ax.bar(preds, obs, width=0.08, alpha=0.6, color="forestgreen", edgecolor="black", label="Observed Rate")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel("Mean P(Referable DR)", fontsize=10)
        ax.set_ylabel("True Referable Proportion", fontsize=10)
        ax.set_title(f"{title}\nBinary ECE = {ece:.4f}", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
