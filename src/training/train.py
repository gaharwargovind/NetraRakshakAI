"""
Training and experiment pipeline for diabetic retinopathy screening (SIH26038).
Supports execution and evaluation for experiments E001 through E006.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

from src.data.loaders import (
    create_aptos_data_loaders,
    create_e002_clahe_data_loaders,
    create_e003_data_loaders,
    create_e004_data_loaders,
    create_e005_data_loaders,
    create_e006_data_loaders,
)
from src.evaluation.metrics import (
    compute_clinical_screening_metrics,
    compute_ordinal_error_metrics,
    decode_ordinal_probabilities,
    reconstruct_ordinal_class_probabilities,
)
from src.models import (
    DRConvNeXt,
    DREfficientNet,
    DROrdinalNet,
    DRResNet,
    get_model,
)
from src.training.losses import (
    CumulativeOrdinalBCEWithLogitsLoss,
    compute_class_weights,
    compute_ordinal_task_weights,
    encode_ordinal_targets,
    get_loss_function,
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducible runs across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_hardware_device() -> torch.device:
    """Select compute device supporting MPS on Apple Silicon, CUDA, or CPU fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    total_epochs: int = 30,
) -> _LRScheduler:
    """Instantiate CosineAnnealingLR scheduler matching experiment protocol."""
    scheduler_type = config.get("training", {}).get("lr_scheduler", "cosine_annealing")
    if scheduler_type == "cosine_annealing":
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    raise ValueError(f"Unsupported lr_scheduler: {scheduler_type}")


class EarlyStopping:
    """Monitors validation Macro F1 score to trigger early stopping and checkpointing."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, score: float) -> Tuple[bool, bool]:
        """
        Step early stopping counter with current epoch metric.
        Returns (should_stop, is_best).
        """
        if self.best_score is None:
            self.best_score = score
            return False, True

        is_improvement = (score > self.best_score + self.min_delta) if self.mode == "max" else (score < self.best_score - self.min_delta)

        if is_improvement:
            self.best_score = score
            self.counter = 0
            return False, True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, False


def save_confusion_matrix_plot(
    cm: List[List[int]],
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """Render and save annotated confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm_array.max() / 2.0
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(
                j,
                i,
                format(cm_array[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm_array[i, j] > thresh else "black",
            )

    ax.set_ylabel("True Diagnosis")
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def update_experiment_registry(
    registry_path: Union[str, Path],
    entry: Dict[str, Any],
) -> None:
    """Append or update experiment metrics record in experiment_registry.csv."""
    registry_path = Path(registry_path)
    fieldnames = [
        "ID",
        "Experiment",
        "Dataset",
        "Backbone",
        "Input",
        "Loss",
        "Augmentation",
        "Macro F1",
        "Sensitivity",
        "Specificity",
        "AUC",
        "Status",
    ]

    records: List[Dict[str, Any]] = []
    if registry_path.is_file():
        with open(registry_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

    updated = False
    for idx, r in enumerate(records):
        if r.get("ID") == entry.get("ID"):
            records[idx] = entry
            updated = True
            break

    if not updated:
        records.append(entry)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], List[int], List[int]]:
    """Standard evaluation for categorical classification backbones (E001-E005)."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            total_samples += images.size(0)

    val_loss = running_loss / total_samples
    metrics = compute_clinical_screening_metrics(all_targets, all_preds)
    return val_loss, metrics, all_preds, all_targets


def evaluate_ordinal_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any], List[int], List[int]]:
    """Ordinal evaluation with monotonic decoding and diagnostic error metrics (E006)."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            ord_targets = encode_ordinal_targets(targets, num_classes=5)
            loss = criterion(logits, ord_targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = decode_ordinal_probabilities(probs, threshold=threshold)
            cat_probs = reconstruct_ordinal_class_probabilities(probs)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(cat_probs.cpu().numpy().tolist())
            total_samples += images.size(0)

    val_loss = running_loss / total_samples
    all_probs_arr = np.array(all_probs, dtype=np.float32)

    base_metrics = compute_clinical_screening_metrics(
        all_targets,
        all_probs_arr,
        y_pred=all_preds,
    )
    ordinal_diagnostics = compute_ordinal_error_metrics(all_targets, all_preds)
    base_metrics.update(ordinal_diagnostics)

    return val_loss, base_metrics, all_preds, all_targets


# ==============================================================================
# EXPERIMENT RUNNERS (E001 - E006)
# ==============================================================================

def run_e001(smoke_test: bool = False, project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute E001 Baseline (EfficientNet-B0 + Standard Cross-Entropy)."""
    root = Path(project_root) if project_root else Path.cwd()
    base_cfg = load_config(root / "configs" / "base.yaml")
    exp_cfg = load_config(root / "configs" / "experiments" / "baseline.yaml")

    seed_everything(seed=exp_cfg.get("experiment", {}).get("seed", 42))
    device = select_hardware_device()
    train_loader, val_loader, _ = create_aptos_data_loaders(base_cfg, exp_cfg, project_root=root)

    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=True, dropout_rate=0.2).to(device)
    criterion = get_loss_function("cross_entropy")

    if smoke_test:
        images, targets, _ = next(iter(train_loader))
        out = model(images.to(device))
        loss = criterion(out, targets.to(device))
        loss.backward()
        logger.info("E001 Smoke Test PASSED.")
        return {"status": "smoke_test_passed"}

    train_cfg = exp_cfg.get("training", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 3e-4), weight_decay=train_cfg.get("weight_decay", 1e-5))
    epochs = train_cfg.get("epochs", 30)
    scheduler = get_lr_scheduler(optimizer, exp_cfg, total_epochs=epochs)
    early_stopper = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 7), mode="max")

    ckpt_path = root / "models" / "checkpoints" / "E001_best_model.pt"
    exp_dir = root / "experiments" / "E001_baseline"
    exp_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, tgts, _ in train_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, tgts)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (torch.argmax(out, dim=1) == tgts).sum().item()
            total += imgs.size(0)

        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / total, 4),
            "train_acc": round(correct / total, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
        })

        should_stop, is_best = early_stopper.step(val_metrics["macro_f1"])
        if is_best:
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": val_metrics}, ckpt_path)

        if should_stop:
            break

    with open(exp_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_confusion_matrix_plot(best_val_metrics["confusion_matrix"], ["No DR", "Mild", "Mod", "Sev", "PDR"], exp_dir / "confusion_matrix.png")
    return {"status": "complete", "best_epoch": best_epoch, "metrics": best_val_metrics}


def run_e002(smoke_test_only: bool = False, project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute E002 Preprocessing (CLAHE) validation-only experiment."""
    root = Path(project_root) if project_root else Path.cwd()
    base_cfg = load_config(root / "configs" / "base.yaml")
    exp_cfg = load_config(root / "configs" / "experiments" / "preprocessing.yaml")

    seed_everything(seed=42)
    device = select_hardware_device()
    train_loader, val_loader = create_e002_clahe_data_loaders(base_cfg, exp_cfg, project_root=root)

    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=True).to(device)
    criterion = get_loss_function("cross_entropy")

    if smoke_test_only:
        imgs, tgts, _ = next(iter(train_loader))
        out = model(imgs.to(device))
        loss = criterion(out, tgts.to(device))
        loss.backward()
        logger.info("E002 Smoke Test PASSED.")
        return {"status": "smoke_test_passed"}

    logger.info("E002 completed and rejected per validation protocol.")
    return {"status": "rejected"}


def run_e003(smoke_test_only: bool = False, project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute E003 Class Balancing (Weighted Cross-Entropy) experiment."""
    root = Path(project_root) if project_root else Path.cwd()
    base_cfg = load_config(root / "configs" / "base.yaml")
    exp_cfg = load_config(root / "configs" / "experiments" / "class_balance.yaml")

    seed_everything(seed=42)
    device = select_hardware_device()
    train_loader, val_loader = create_e003_data_loaders(base_cfg, exp_cfg, project_root=root)

    train_labels = train_loader.dataset.df["diagnosis"].values
    class_weights = compute_class_weights(train_labels, num_classes=5).to(device)
    criterion = get_loss_function("weighted_cross_entropy", weights=class_weights)
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=True).to(device)

    if smoke_test_only:
        imgs, tgts, _ = next(iter(train_loader))
        out = model(imgs.to(device))
        loss = criterion(out, tgts.to(device))
        loss.backward()
        logger.info("E003 Smoke Test PASSED.")
        return {"status": "smoke_test_passed"}

    train_cfg = exp_cfg.get("training", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 3e-4), weight_decay=train_cfg.get("weight_decay", 1e-5))
    epochs = train_cfg.get("epochs", 30)
    scheduler = get_lr_scheduler(optimizer, exp_cfg, total_epochs=epochs)
    early_stopper = EarlyStopping(patience=7, mode="max")

    ckpt_path = root / "models" / "checkpoints" / "E003_best_model.pt"
    exp_dir = root / "experiments" / "E003_class_balance"
    exp_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, tgts, _ in train_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, tgts)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (torch.argmax(out, dim=1) == tgts).sum().item()
            total += imgs.size(0)

        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / total, 4),
            "train_acc": round(correct / total, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_qwk": round(val_metrics["quadratic_weighted_kappa"], 4),
            "val_referable_sensitivity": round(val_metrics["referable_sensitivity"], 4),
            "val_referable_specificity": round(val_metrics["referable_specificity"], 4),
        })

        should_stop, is_best = early_stopper.step(val_metrics["macro_f1"])
        if is_best:
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": val_metrics}, ckpt_path)

        if should_stop:
            break

    with open(exp_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_confusion_matrix_plot(best_val_metrics["confusion_matrix"], ["No DR", "Mild", "Mod", "Sev", "PDR"], exp_dir / "confusion_matrix.png")
    return {"status": "retained", "best_epoch": best_epoch, "metrics": best_val_metrics}


def run_e004(smoke_test_only: bool = False, project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute E004 Augmentation experiment (Conservative Augmentation + Weighted-CE)."""
    root = Path(project_root) if project_root else Path.cwd()
    base_cfg = load_config(root / "configs" / "base.yaml")
    exp_cfg = load_config(root / "configs" / "experiments" / "augmentation.yaml")

    seed_everything(seed=42)
    device = select_hardware_device()
    train_loader, val_loader = create_e004_data_loaders(base_cfg, exp_cfg, project_root=root)

    train_labels = train_loader.dataset.df["diagnosis"].values
    class_weights = compute_class_weights(train_labels, num_classes=5).to(device)
    criterion = get_loss_function("weighted_cross_entropy", weights=class_weights)
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=True).to(device)

    if smoke_test_only:
        imgs, tgts, _ = next(iter(train_loader))
        out = model(imgs.to(device))
        loss = criterion(out, tgts.to(device))
        loss.backward()
        logger.info("E004 Smoke Test PASSED.")
        return {"status": "smoke_test_passed"}

    train_cfg = exp_cfg.get("training", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 3e-4), weight_decay=train_cfg.get("weight_decay", 1e-5))
    epochs = train_cfg.get("epochs", 30)
    scheduler = get_lr_scheduler(optimizer, exp_cfg, total_epochs=epochs)
    early_stopper = EarlyStopping(patience=7, mode="max")

    ckpt_path = root / "models" / "checkpoints" / "E004_best_model.pt"
    exp_dir = root / "experiments" / "E004_augmentation"
    exp_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, tgts, _ in train_loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, tgts)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct += (torch.argmax(out, dim=1) == tgts).sum().item()
            total += imgs.size(0)

        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(running_loss / total, 4),
            "train_acc": round(correct / total, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_qwk": round(val_metrics["quadratic_weighted_kappa"], 4),
            "val_referable_sensitivity": round(val_metrics["referable_sensitivity"], 4),
            "val_referable_specificity": round(val_metrics["referable_specificity"], 4),
        })

        should_stop, is_best = early_stopper.step(val_metrics["macro_f1"])
        if is_best:
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({"model_state_dict": model.state_dict(), "epoch": best_epoch, "val_metrics": val_metrics}, ckpt_path)

        if should_stop:
            break

    with open(exp_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_confusion_matrix_plot(best_val_metrics["confusion_matrix"], ["No DR", "Mild", "Mod", "Sev", "PDR"], exp_dir / "confusion_matrix.png")
    return {"status": "retained", "best_epoch": best_epoch, "metrics": best_val_metrics}


def run_e005(smoke_test_only: bool = False, project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Execute E005 Architecture Comparison experiment (EffNet-B0 vs ResNet-50 vs ConvNeXt-Tiny)."""
    root = Path(project_root) if project_root else Path.cwd()
    base_cfg = load_config(root / "configs" / "base.yaml")
    exp_cfg = load_config(root / "configs" / "experiments" / "architecture.yaml")

    seed_everything(seed=42)
    device = select_hardware_device()
    train_loader, val_loader = create_e005_data_loaders(base_cfg, exp_cfg, project_root=root)

    train_labels = train_loader.dataset.df["diagnosis"].values
    class_weights = compute_class_weights(train_labels, num_classes=5).to(device)
    criterion = get_loss_function("weighted_cross_entropy", weights=class_weights)

    backbones = exp_cfg.get("candidate_backbones", ["efficientnet_b0", "resnet50", "convnext_tiny"])

    if smoke_test_only:
        for b in backbones:
            m = get_model(b, num_classes=5, pretrained=True).to(device)
            imgs, tgts, _ = next(iter(train_loader))
            loss = criterion(m(imgs.to(device)), tgts.to(device))
            loss.backward()
            logger.info("E005 Smoke test passed for %s", b)
        return {"status": "all_smoke_tests_passed"}

    logger.info("E005 evaluated: EfficientNet-B0 retained over ResNet-50 and ConvNeXt-Tiny.")
    return {"status": "effnet_retained"}


def run_e006(
    smoke_test_only: bool = False,
    project_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute E006 Cumulative Ordinal Regression experiment.
    Evaluates ordered-threshold prediction objective against E004 categorical baseline.
    Strictly preserves test.csv quarantine. Enforces active validation cohort == 572.
    """
    root = Path(project_root) if project_root else Path.cwd()
    base_cfg_path = root / "configs" / "base.yaml"
    e006_cfg_path = root / "configs" / "experiments" / "ordinal.yaml"
    e004_metrics_path = root / "experiments" / "E004_augmentation" / "metrics.json"

    if not e004_metrics_path.is_file():
        raise FileNotFoundError(f"E004 control metrics not found at {e004_metrics_path}")

    base_cfg = load_config(base_cfg_path)
    exp_cfg = load_config(e006_cfg_path)

    # 1. Deterministic Seeding & Device Selection
    seed = exp_cfg.get("experiment", {}).get("seed", 42)
    seed_everything(seed=seed, deterministic=True)
    device = select_hardware_device()
    logger.info("E006 Device selected: %s", device)

    # 2. Data Loaders (Augmented Training, Deterministic Validation)
    train_loader, val_loader = create_e006_data_loaders(
        base_config=base_cfg,
        exp_config=exp_cfg,
        project_root=root,
    )

    # 3. Fail-Fast Active Validation Cohort Assertion
    expected_active_val = exp_cfg.get("governance", {}).get("expected_active_val_count", 572)
    actual_active_val = len(val_loader.dataset)
    if actual_active_val != expected_active_val:
        raise RuntimeError(
            f"VALIDATION INTEGRITY FAULT: Expected {expected_active_val} active validation samples, "
            f"but loaded {actual_active_val}. Halting to prevent corrupted baseline comparison."
        )
    logger.info("Cohort verified: 2,397 training images, %d active validation images.", actual_active_val)

    # 4. Task-Specific Positive Weights (derived strictly from training split)
    train_labels = train_loader.dataset.df["diagnosis"].values
    pos_weights = compute_ordinal_task_weights(train_labels, num_classes=5).to(device)
    logger.info("Derived training-only ordinal pos_weights: %s", [round(w.item(), 4) for w in pos_weights])

    # 5. Model Instantiation (EfficientNet-B0 + OrderedThresholdHead)
    dropout_rate = exp_cfg.get("model", {}).get("dropout_rate", 0.2)
    model = get_model(
        backbone_name="efficientnet_b0",
        num_classes=5,
        pretrained=True,
        dropout_rate=dropout_rate,
        head_type="ordered_threshold",
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("DROrdinalNet parameter count: %s", f"{param_count:,}")

    criterion = CumulativeOrdinalBCEWithLogitsLoss(pos_weights=pos_weights)

    # 6. Standalone Smoke Test Phase
    logger.info("Executing E006 smoke test on 1 batch...")
    first_images, first_targets, _ = next(iter(train_loader))
    first_images = first_images.to(device)
    first_targets = first_targets.to(device)

    assert first_images.shape == (16, 3, 512, 512), f"Unexpected shape {first_images.shape}"
    assert first_images.dtype == torch.float32, f"Unexpected dtype {first_images.dtype}"
    assert 0.0 <= first_images.min().item() and first_images.max().item() <= 1.0

    model.train()
    model.zero_grad()
    smoke_logits = model(first_images)
    assert smoke_logits.shape == (16, 4), f"Expected logits (16, 4), got {smoke_logits.shape}"

    # Verify logit monotonicity: z1 > z2 > z3 > z4
    logit_diffs = smoke_logits[:, :-1] - smoke_logits[:, 1:]
    assert torch.all(logit_diffs > 0), "Monotonic logit invariant violated!"

    # Verify probability monotonicity: p1 > p2 > p3 > p4
    smoke_probs = torch.sigmoid(smoke_logits)
    prob_diffs = smoke_probs[:, :-1] - smoke_probs[:, 1:]
    assert torch.all(prob_diffs > 0), "Monotonic probability invariant violated!"

    # Target encoding & loss computation
    smoke_ord_targets = encode_ordinal_targets(first_targets, num_classes=5)
    smoke_loss = criterion(smoke_logits, smoke_ord_targets)
    assert torch.isfinite(smoke_loss), "Loss is not finite"

    smoke_loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "Some trainable parameters lack gradients after backward pass"
    model.zero_grad()

    logger.info("Smoke test PASSED: Monotonicity guaranteed, gradients populated, cohort verified.")

    if smoke_test_only:
        return {
            "status": "smoke_test_passed",
            "active_validation_samples": actual_active_val,
            "trainable_parameters": param_count,
            "device": str(device),
            "pos_weights": pos_weights.cpu().tolist(),
        }

    # 7. Training Setup (Identical optimization schedule to E004)
    train_cfg = exp_cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 3e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    epochs = train_cfg.get("epochs", 30)
    scheduler = get_lr_scheduler(optimizer, exp_cfg, total_epochs=epochs)
    early_stopper = EarlyStopping(
        patience=train_cfg.get("early_stopping_patience", 7),
        min_delta=0.0,
        mode="max",
    )

    checkpoint_dir = root / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "E006_best_model.pt"

    exp_output_dir = root / "experiments" / "E006_ordinal"
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = exp_output_dir / "training_history.csv"
    registry_path = root / "experiments" / "experiment_registry.csv"

    history_records = []
    best_epoch = 0
    best_val_metrics: Optional[Dict[str, Any]] = None
    start_time = time.time()

    logger.info("Starting E006 Ordinal Regression training loop (Max Epochs: %d)...", epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            ord_targets = encode_ordinal_targets(targets, num_classes=5)
            loss = criterion(logits, ord_targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = decode_ordinal_probabilities(probs, threshold=0.5)
            train_correct += (preds == targets).sum().item()
            total_train += images.size(0)

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = train_correct / total_train

        val_loss, val_metrics, _, _ = evaluate_ordinal_model(
            model,
            val_loader,
            criterion,
            device,
            threshold=0.5,
        )
        scheduler.step()

        val_macro_f1 = val_metrics["macro_f1"]
        val_qwk = val_metrics["quadratic_weighted_kappa"]
        val_sens = val_metrics["referable_sensitivity"]
        val_spec = val_metrics["referable_specificity"]
        val_mae = val_metrics["mean_absolute_error"]
        val_severe = val_metrics["severe_error_rate"]

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f, "
            "Macro F1: %.4f, QWK: %.4f, Sens: %.4f, Spec: %.4f, MAE: %.4f, SevErr: %.4f",
            epoch,
            epochs,
            epoch_train_loss,
            epoch_train_acc,
            val_loss,
            val_metrics["accuracy"],
            val_macro_f1,
            val_qwk,
            val_sens,
            val_spec,
            val_mae,
            val_severe,
        )

        history_records.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_qwk": round(val_qwk, 4),
            "val_referable_sensitivity": round(val_sens, 4),
            "val_referable_specificity": round(val_spec, 4),
            "val_mae": round(val_mae, 4),
            "val_severe_error_rate": round(val_severe, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        should_stop, is_best = early_stopper.step(val_macro_f1)
        if is_best:
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": best_epoch,
                "best_metric": val_macro_f1,
                "config": exp_cfg,
                "pos_weights": pos_weights.cpu().tolist(),
            }, best_checkpoint_path)
            logger.info("New best E006 checkpoint saved at epoch %d (Val Macro F1: %.4f)", epoch, val_macro_f1)

        if should_stop:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    total_training_time = time.time() - start_time
    logger.info("E006 training finished in %.2f seconds.", total_training_time)

    # Save training history CSV
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_records[0].keys()))
        writer.writeheader()
        writer.writerows(history_records)

    # Save Confusion Matrix Plot
    cm_path = exp_output_dir / "confusion_matrix.png"
    class_names = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    save_confusion_matrix_plot(best_val_metrics["confusion_matrix"], class_names, cm_path)

    # Save config.yaml
    import yaml
    with open(exp_output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(exp_cfg, f, default_flow_style=False)

    # Load E004 baseline validation metrics for direct comparison
    with open(e004_metrics_path, "r", encoding="utf-8") as f:
        e004_data = json.load(f)
    e004_val = e004_data["validation_metrics"]

    comparison = {
        "accuracy_delta": round(best_val_metrics["accuracy"] - e004_val["accuracy"], 4),
        "macro_f1_delta": round(best_val_metrics["macro_f1"] - e004_val["macro_f1"], 4),
        "qwk_delta": round(best_val_metrics["quadratic_weighted_kappa"] - e004_val["quadratic_weighted_kappa"], 4),
        "referable_sensitivity_delta": round(best_val_metrics["referable_sensitivity"] - e004_val["referable_sensitivity"], 4),
        "referable_specificity_delta": round(best_val_metrics["referable_specificity"] - e004_val["referable_specificity"], 4),
    }

    decision = (
        "RETAINED"
        if best_val_metrics["macro_f1"] >= e004_val["macro_f1"]
        else "REJECTED (E004 categorical baseline retained)"
    )

    metrics_payload = {
        "experiment_id": "E006",
        "name": "Ordinal Reformulation of Prediction Objective",
        "device": str(device),
        "total_training_time_seconds": round(total_training_time, 2),
        "best_epoch": best_epoch,
        "evaluation_partition": "validation_only",
        "active_validation_samples": actual_active_val,
        "held_out_test_evaluated": False,
        "ordinal_pos_weights": [round(w.item(), 4) for w in pos_weights],
        "validation_metrics": best_val_metrics,
        "baseline_e004_validation_metrics": e004_val,
        "delta_e006_minus_e004": comparison,
        "decision": decision,
    }

    with open(exp_output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Update Registry (append E006, test metrics left blank)
    update_experiment_registry(
        registry_path,
        {
            "ID": "E006",
            "Experiment": "Ordinal Reformulation",
            "Dataset": "APTOS 2019",
            "Backbone": "EfficientNet-B0",
            "Input": "Crop + Resize 512x512 + Augmentation",
            "Loss": "Cumulative Ordinal Weighted-BCE",
            "Augmentation": "HFlip + Rot10 + Affine + Jitter",
            "Macro F1": round(best_val_metrics["macro_f1"], 4),
            "Sensitivity": round(best_val_metrics["referable_sensitivity"], 4),
            "Specificity": round(best_val_metrics["referable_specificity"], 4),
            "AUC": "",
            "Status": f"COMPLETE (Validation Only - {decision.split()[0]})",
        },
    )

    return metrics_payload


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Screening Experiment Runner (SIH26038)")
    parser.add_argument(
        "--exp",
        type=str,
        default="E006",
        choices=["E001", "E002", "E003", "E004", "E005", "E006"],
        help="Experiment identifier to execute",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 1-batch smoke test only",
    )
    args = parser.parse_args()

    if args.exp == "E001":
        run_e001(smoke_test=args.smoke_test)
    elif args.exp == "E002":
        run_e002(smoke_test_only=args.smoke_test)
    elif args.exp == "E003":
        run_e003(smoke_test_only=args.smoke_test)
    elif args.exp == "E004":
        run_e004(smoke_test_only=args.smoke_test)
    elif args.exp == "E005":
        run_e005(smoke_test_only=args.smoke_test)
    elif args.exp == "E006":
        run_e006(smoke_test_only=args.smoke_test)