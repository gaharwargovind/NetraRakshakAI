import os
import sys
import json
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.ordinal import DROrdinalNet

try:
    from src.preprocessing.crop_retina import crop_retina_circle
except ImportError:
    try:
        from src.data.transforms import crop_retina_circle
    except ImportError:
        def crop_retina_circle(image: np.ndarray, tol: int = 10) -> np.ndarray:
            if image.ndim == 3:
                gray = np.max(image, axis=2)
                mask = gray > tol
                if not np.any(mask):
                    return image
                check_rows = mask.any(1)
                check_cols = mask.any(0)
                r0, r1 = np.where(check_rows)[0][[0, -1]]
                c0, c1 = np.where(check_cols)[0][[0, -1]]
                return image[r0:r1 + 1, c0:c1 + 1]
            return image

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def process_retina_crop(img_rgb: Image.Image) -> Image.Image:
    img_np = np.array(img_rgb)
    try:
        out = crop_retina_circle(img_np)
        if isinstance(out, np.ndarray) and out.size > 0:
            return Image.fromarray(out)
        elif isinstance(out, Image.Image):
            return out
    except Exception:
        pass
    return img_rgb

def plot_and_save_cm(cm: np.ndarray, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ticks = np.arange(5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"Grade {i}" for i in range(5)])
    ax.set_yticklabels([f"Grade {i}" for i in range(5)])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted Grade", fontsize=10)
    ax.set_ylabel("True Grade", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

class E006ValidationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        p = self.img_dir / f"{self.ids[idx]}.png"
        img = Image.open(p).convert("RGB")
        return self.transform(img), self.labels[idx]

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device selected: {device}")

    protected_files = [
        ROOT_DIR / "experiments/E001_baseline/metrics.json",
        ROOT_DIR / "experiments/E002_preprocessing/metrics.json",
        ROOT_DIR / "experiments/E003_class_balance/metrics.json",
        ROOT_DIR / "experiments/E004_augmentation/metrics.json",
        ROOT_DIR / "experiments/E005_architecture/metrics.json",
        ROOT_DIR / "experiments/E006_ordinal/metrics.json",
        ROOT_DIR / "models/checkpoints/E006_best_model.pt",
    ]
    pre_hashes = {p: sha256_file(p) for p in protected_files if p.is_file()}

    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    images_dir = ROOT_DIR / "data/raw/aptos/train_images"

    assert val_csv.is_file(), f"Missing {val_csv}"
    val_df = pd.read_csv(val_csv)
    assert len(val_df) == 601, f"Expected 601, got {len(val_df)}"
    print("Validation cohort verified: 601 samples")

    ckpt_path = ROOT_DIR / "models/checkpoints/E006_best_model.pt"
    assert ckpt_path.is_file(), f"Missing {ckpt_path}"

    model = DROrdinalNet(backbone_name="efficientnet_b0", num_classes=5)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    val_transform = T.Compose([
        T.Lambda(process_retina_crop),
        T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = E006ValidationDataset(val_df, images_dir, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_targets, all_preds, all_cum, all_nat = [], [], [], []
    print("Evaluating 601 validation images...")
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            cum_probs = torch.sigmoid(model(imgs))
            preds = (cum_probs > 0.5).sum(dim=-1).long()

            p1 = cum_probs[:, 0:1]
            p2 = cum_probs[:, 1:2]
            p3 = cum_probs[:, 2:3]
            p4 = cum_probs[:, 3:4]
            nat = torch.cat([1.0 - p1, p1 - p2, p2 - p3, p3 - p4, p4], dim=1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_cum.append(cum_probs.cpu().numpy())
            all_nat.append(nat.cpu().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    cum_arr = np.vstack(all_cum)
    nat_arr = np.vstack(all_nat)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    ref_true = (y_true >= 2).astype(int)
    ref_pred = (y_pred >= 2).astype(int)
    tp = int(np.sum((ref_true == 1) & (ref_pred == 1)))
    fn = int(np.sum((ref_true == 1) & (ref_pred == 0)))
    tn = int(np.sum((ref_true == 0) & (ref_pred == 0)))
    fp = int(np.sum((ref_true == 0) & (ref_pred == 1)))
    ref_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ref_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0)
    per_class = {str(i): {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4), "support": int(supp[i])} for i in range(5)}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    adj_err = float(np.mean(np.abs(y_true - y_pred) == 1))
    sev_err = float(np.mean(np.abs(y_true - y_pred) >= 2))

    try:
        macro_auc = float(roc_auc_score(np.eye(5)[y_true], nat_arr, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = None

    try:
        ref_auc = float(roc_auc_score(ref_true, cum_arr[:, 1]))
    except Exception:
        ref_auc = None

    out_dir = ROOT_DIR / "experiments/E006_current_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "experiment_id": "E006_current_validation",
        "evaluation_type": "corrected_reevaluation",
        "validation_manifest": "data/processed/aptos/validation.csv",
        "validation_samples": 601,
        "checkpoint_used": str(ckpt_path),
        "preprocessing": "RGB -> crop_retina_circle -> resize 512x512 -> ToTensor -> ImageNet normalize",
        "decoding_threshold": 0.5,
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "quadratic_weighted_kappa": round(qwk, 4),
            "referable_sensitivity": round(ref_sens, 4),
            "referable_specificity": round(ref_spec, 4),
            "mae": round(mae, 4),
            "adjacent_error_rate": round(adj_err, 4),
            "severe_error_rate": round(sev_err, 4),
            "macro_roc_auc": round(macro_auc, 4) if macro_auc is not None else None,
            "referable_roc_auc": round(ref_auc, 4) if ref_auc is not None else None,
            "per_class": per_class,
            "confusion_matrix": cm,
        }
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    plot_and_save_cm(np.array(cm), "E006 Current Validation Confusion Matrix (N=601)", out_dir / "confusion_matrix.png")

    readme_lines = [
        "# E006 Corrected Re-Evaluation (Current Validation Manifest)",
        "",
        "## Disclosures",
        "- Corrected reevaluation on current 601 validation records.",
        "- Historical 572-image E004 cohort is unrecoverable.",
        "- E004 historical metrics are NOT directly comparable.",
        "- Previous common-cohort evaluation was invalid due to raw resize domain shift.",
        "- Data partition isolation: test partition was not accessed.",
        "- Zero training policy: no retraining or fine-tuning occurred.",
        "",
        f"- Accuracy: {acc:.4f}",
        f"- Macro F1: {macro_f1:.4f}",
        f"- Weighted F1: {weighted_f1:.4f}",
        f"- QWK: {qwk:.4f}",
        f"- Referable Sensitivity: {ref_sens:.4f}",
        f"- Referable Specificity: {ref_spec:.4f}",
        f"- MAE: {mae:.4f}",
        f"- Severe Error Rate: {sev_err:.4f}",
        "",
        "Confusion Matrix:",
        str(np.array(cm)),
    ]
    with open(out_dir / "README.md", "w") as f:
        f.write("\n".join(readme_lines) + "\n")

    for p, orig_h in pre_hashes.items():
        assert sha256_file(p) == orig_h, f"INTEGRITY VIOLATION: {p} modified"

    sep = "=" * 70
    print()
    print(sep)
    print("      E006 CORRECTED VALIDATION RE-EVALUATION REPORT")
    print(sep)
    print(f"- Validation Sample Count: {len(val_df)}")
    print(f"- Checkpoint Used:         {ckpt_path}")
    print("- Preprocessing:           RGB -> crop_retina_circle -> Resize(512,512) -> ToTensor -> Normalize")
    print(f"- Accuracy:                {acc:.4f}")
    print(f"- Macro F1:                {macro_f1:.4f}")
    print(f"- Weighted F1:             {weighted_f1:.4f}")
    print(f"- QWK:                     {qwk:.4f}")
    print(f"- Referable Sensitivity:   {ref_sens:.4f}")
    print(f"- Referable Specificity:   {ref_spec:.4f}")
    print(f"- MAE:                     {mae:.4f}")
    print(f"- Adjacent Error Rate:     {adj_err:.4f}")
    print(f"- Severe Error Rate:       {sev_err:.4f}")
    if macro_auc is not None:
        print(f"- Macro ROC-AUC:           {macro_auc:.4f}")
    if ref_auc is not None:
        print(f"- Referable ROC-AUC:       {ref_auc:.4f}")
    print()
    print("Confusion Matrix (Rows: True, Columns: Predicted):")
    print(np.array(cm))
    print(sep)

if __name__ == "__main__":
    main()
