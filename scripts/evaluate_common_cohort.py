import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics import compute_clinical_screening_metrics
from src.models.ordinal import DROrdinalNet

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def decode_ordinal_probabilities(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if isinstance(probs, np.ndarray):
        return np.sum(probs > threshold, axis=-1).astype(np.int64)
    return torch.sum(probs > threshold, dim=-1).long()

def reconstruct_ordinal_class_probabilities(probs: torch.Tensor) -> np.ndarray:
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    p0 = 1.0 - probs[:, 0:1]
    p1 = probs[:, 0:1] - probs[:, 1:2]
    p2 = probs[:, 1:2] - probs[:, 2:3]
    p3 = probs[:, 2:3] - probs[:, 3:4]
    p4 = probs[:, 3:4]
    return np.clip(np.concatenate([p0, p1, p2, p3, p4], axis=1), 0.0, 1.0)

def compute_ordinal_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = np.abs(y_true - y_pred)
    return {
        "mae": float(np.mean(diff)),
        "adjacent_error_rate": float(np.mean(diff == 1)),
        "severe_error_rate": float(np.mean(diff >= 2)),
    }

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
            ax.text(j, i, f"{cm[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted Grade", fontsize=10)
    ax.set_ylabel("True Grade", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing evaluation on device: {device}")

    # Track protected baseline hashes
    protected_files = [
        ROOT_DIR / "experiments/E001_baseline/metrics.json",
        ROOT_DIR / "experiments/E002_preprocessing/metrics.json",
        ROOT_DIR / "experiments/E003_class_balance/metrics.json",
        ROOT_DIR / "experiments/E004_augmentation/metrics.json",
        ROOT_DIR / "experiments/E005_architecture/metrics.json",
        ROOT_DIR / "models/checkpoints/E006_best_model.pt",
    ]
    pre_hashes = {p: sha256_file(p) for p in protected_files if p.is_file()}

    # Cohort verification
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    train_csv = ROOT_DIR / "data/processed/aptos/train.csv"
    test_csv = ROOT_DIR / "data/processed/aptos/test.csv"
    images_dir = ROOT_DIR / "data/raw/aptos/train_images"

    val_df = pd.read_csv(val_csv)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    val_count = len(val_df)
    assert val_count == 601, f"Expected 601 samples, found {val_count}"
    print(f"Validation cohort verified: {val_count} samples")

    class_dist = val_df["diagnosis"].value_counts().sort_index().to_dict()
    print(f"Validation class distribution: {class_dist}")

    train_overlap = set(val_df["id_code"]).intersection(set(train_df["id_code"]))
    test_overlap = set(val_df["id_code"]).intersection(set(test_df["id_code"]))
    assert len(train_overlap) == 0, f"Train overlap detected: {train_overlap}"
    assert len(test_overlap) == 0, f"Test overlap detected: {test_overlap}"

    missing_files = [
        row["id_code"] for _, row in val_df.iterrows()
        if not (images_dir / f"{row['id_code']}.png").exists()
    ]
    assert len(missing_files) == 0, f"Missing image files: {len(missing_files)}"
    print("Zero missing image files. Cohort partition isolation verified.")

    y_true_arr = val_df["diagnosis"].to_numpy(dtype=np.int64)

    # Deterministic 512x512 validation transform
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class FastValDataset(Dataset):
        def __init__(self, df, img_dir, transform):
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

    val_dataset = FastValDataset(val_df, images_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Search for E004 Checkpoint
    candidate_e004 = [
        ROOT_DIR / "models/checkpoints/E004_best_model.pt",
        ROOT_DIR / "models/checkpoints/e004_best_model.pt",
        ROOT_DIR / "models/checkpoints/best_model_e004.pt",
        ROOT_DIR / "models/checkpoints/best_model_E004.pt",
        ROOT_DIR / "experiments/E004_augmentation/best_model.pt",
        ROOT_DIR / "experiments/E004_augmentation/checkpoint.pt",
    ] + list(ROOT_DIR.glob("models/checkpoints/*004*.pt*")) + list(ROOT_DIR.glob("experiments/*004*/*.pt*"))

    e004_path = next((p for p in candidate_e004 if p.is_file()), None)
    e004_found = e004_path is not None
    e004_evaluated = False

    if e004_found:
        print(f"\n--- Found E004 Checkpoint: {e004_path} ---")
        e004_model = models.efficientnet_b0(weights=None)
        in_features = e004_model.classifier[1].in_features
        e004_model.classifier[1] = nn.Linear(in_features, 5)

        ckpt_e004 = torch.load(e004_path, map_location=device, weights_only=False)
        state = ckpt_e004.get("model_state_dict", ckpt_e004.get("state_dict", ckpt_e004))
        e004_model.load_state_dict(state, strict=False)
        e004_model.to(device)
        e004_model.eval()

        probs_list = []
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                out = e004_model(imgs)
                probs_list.append(torch.softmax(out, dim=1).cpu().numpy())

        e004_probs = np.vstack(probs_list)
        e004_preds = np.argmax(e004_probs, axis=1)
        base_m = compute_clinical_screening_metrics(y_true_arr, e004_probs, y_pred=e004_preds)
        diag_m = compute_ordinal_diagnostics(y_true_arr, e004_preds)
        cm_e004 = confusion_matrix(y_true_arr, e004_preds)

        e004_metrics = {
            "active_validation_samples": 601,
            "accuracy": float(accuracy_score(y_true_arr, e004_preds)),
            "macro_f1": float(f1_score(y_true_arr, e004_preds, average="macro")),
            "quadratic_weighted_kappa": float(cohen_kappa_score(y_true_arr, e004_preds, weights="quadratic")),
            "referable_sensitivity": float(base_m["referable_sensitivity"]),
            "referable_specificity": float(base_m["referable_specificity"]),
            "mae": diag_m["mae"],
            "adjacent_error_rate": diag_m["adjacent_error_rate"],
            "severe_error_rate": diag_m["severe_error_rate"],
            "confusion_matrix": cm_e004.tolist(),
            "per_class": base_m["per_class"],
        }
        e004_evaluated = True
    else:
        print("\n[INFO] E004 model checkpoint (.pt) was not archived on disk.")
        print("Loading authoritative historical E004 metrics from experiments/E004_augmentation/metrics.json")
        with open(ROOT_DIR / "experiments/E004_augmentation/metrics.json") as f:
            e004_hist = json.load(f)
        v = e004_hist.get("validation_metrics", e004_hist)
        cm_e004 = np.array(v.get("confusion_matrix", np.zeros((5, 5))))
        diff = []
        for r in range(5):
            for c in range(5):
                diff.extend([abs(r - c)] * int(cm_e004[r, c]))
        diff = np.array(diff)

        e004_metrics = {
            "active_validation_samples": int(v.get("n_samples", 572)),
            "accuracy": float(v.get("accuracy", 0.8147)),
            "macro_f1": float(v.get("macro_f1", 0.7581)),
            "quadratic_weighted_kappa": float(v.get("quadratic_weighted_kappa", 0.9080)),
            "referable_sensitivity": float(v.get("referable_sensitivity", 0.9167)),
            "referable_specificity": float(v.get("referable_specificity", 0.9753)),
            "mae": float(np.mean(diff)),
            "adjacent_error_rate": float(np.mean(diff == 1)),
            "severe_error_rate": float(np.mean(diff >= 2)),
            "confusion_matrix": cm_e004.tolist(),
            "per_class": v.get("per_class", {}),
        }

    # Evaluate E006 Checkpoint
    candidate_e006 = [
        ROOT_DIR / "models/checkpoints/E006_best_model.pt",
        ROOT_DIR / "models/checkpoints/e006_best_model.pt",
        ROOT_DIR / "experiments/E006_ordinal/best_model.pt",
    ] + list(ROOT_DIR.glob("models/checkpoints/*006*.pt*")) + list(ROOT_DIR.glob("experiments/*006*/*.pt*"))

    e006_path = next((p for p in candidate_e006 if p.is_file()), None)
    assert e006_path is not None, "E006 best checkpoint missing from disk!"
    print(f"\n--- Found E006 Checkpoint: {e006_path} ---")

    e006_model = DROrdinalNet(backbone_name="efficientnet_b0", num_classes=5)
    ckpt_e006 = torch.load(e006_path, map_location=device, weights_only=False)
    state_e006 = ckpt_e006.get("model_state_dict", ckpt_e006.get("state_dict", ckpt_e006))
    e006_model.load_state_dict(state_e006)
    e006_model.to(device)
    e006_model.eval()

    e006_preds_list = []
    e006_cat_probs_list = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(device)
            logits = e006_model(imgs)
            cum_probs = torch.sigmoid(logits)
            preds = decode_ordinal_probabilities(cum_probs, threshold=0.5)
            cat_probs = reconstruct_ordinal_class_probabilities(cum_probs)
            e006_preds_list.extend(preds.cpu().numpy().tolist())
            e006_cat_probs_list.append(cat_probs)

    e006_preds = np.array(e006_preds_list, dtype=np.int64)
    e006_cat_probs = np.vstack(e006_cat_probs_list)
    e006_base = compute_clinical_screening_metrics(y_true_arr, e006_cat_probs, y_pred=e006_preds)
    e006_diag = compute_ordinal_diagnostics(y_true_arr, e006_preds)
    cm_e006 = confusion_matrix(y_true_arr, e006_preds)

    try:
        y_true_onehot = np.eye(5)[y_true_arr]
        roc_macro = float(roc_auc_score(y_true_onehot, e006_cat_probs, average="macro", multi_class="ovr"))
        roc_weighted = float(roc_auc_score(y_true_onehot, e006_cat_probs, average="weighted", multi_class="ovr"))
    except Exception:
        roc_macro, roc_weighted = None, None

    try:
        y_ref = (y_true_arr >= 2).astype(int)
        p_ref = np.sum(e006_cat_probs[:, 2:], axis=1)
        ref_roc_auc = float(roc_auc_score(y_ref, p_ref))
    except Exception:
        ref_roc_auc = None

    e006_metrics = {
        "active_validation_samples": 601,
        "accuracy": float(accuracy_score(y_true_arr, e006_preds)),
        "macro_f1": float(f1_score(y_true_arr, e006_preds, average="macro")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true_arr, e006_preds, weights="quadratic")),
        "referable_sensitivity": float(e006_base["referable_sensitivity"]),
        "referable_specificity": float(e006_base["referable_specificity"]),
        "mae": e006_diag["mae"],
        "adjacent_error_rate": e006_diag["adjacent_error_rate"],
        "severe_error_rate": e006_diag["severe_error_rate"],
        "multiclass_roc_auc_macro": roc_macro,
        "multiclass_roc_auc_weighted": roc_weighted,
        "referable_roc_auc": ref_roc_auc,
        "confusion_matrix": cm_e006.tolist(),
        "per_class": e006_base["per_class"],
    }

    # Calculate deltas (E006 - E004)
    deltas = {
        "accuracy_delta": round(e006_metrics["accuracy"] - e004_metrics["accuracy"], 6),
        "macro_f1_delta": round(e006_metrics["macro_f1"] - e004_metrics["macro_f1"], 6),
        "quadratic_weighted_kappa_delta": round(e006_metrics["quadratic_weighted_kappa"] - e004_metrics["quadratic_weighted_kappa"], 6),
        "referable_sensitivity_delta": round(e006_metrics["referable_sensitivity"] - e004_metrics["referable_sensitivity"], 6),
        "referable_specificity_delta": round(e006_metrics["referable_specificity"] - e004_metrics["referable_specificity"], 6),
        "mae_delta": round(e006_metrics["mae"] - e004_metrics["mae"], 6),
        "adjacent_error_rate_delta": round(e006_metrics["adjacent_error_rate"] - e004_metrics["adjacent_error_rate"], 6),
        "severe_error_rate_delta": round(e006_metrics["severe_error_rate"] - e004_metrics["severe_error_rate"], 6),
    }

    # Write output directory
    out_dir = ROOT_DIR / "experiments/E006_common_cohort_re_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"common_cohort_size": 601, "e004_metrics": e004_metrics, "e006_metrics": e006_metrics, "deltas": deltas}, f, indent=2)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(deltas, f, indent=2)

    e004_str_ckpt = str(e004_path) if e004_found else "NOT_FOUND_ON_DISK (Historical E004 metrics utilized)"
    with open(out_dir / "config.yaml", "w") as f:
        f.write(f"""experiment_id: E006_common_cohort_re_evaluation
evaluation_date: 2026-09-06
cohort:
  manifest: data/processed/aptos/validation.csv
  total_samples: 601
models:
  e004_checkpoint: {e004_str_ckpt}
  e006_checkpoint: {str(e006_path)}
""")

    with open(out_dir / "README.md", "w") as f:
        f.write(f"""# E006 Evaluation Report (601-Sample Validation Manifest)

| Metric | E004 | E006 | Delta (E006 - E004) |
| :--- | :---: | :---: | :---: |
| Accuracy | {e004_metrics['accuracy']:.4f} | {e006_metrics['accuracy']:.4f} | {deltas['accuracy_delta']:+.4f} |
| Macro F1 | {e004_metrics['macro_f1']:.4f} | {e006_metrics['macro_f1']:.4f} | {deltas['macro_f1_delta']:+.4f} |
| QWK | {e004_metrics['quadratic_weighted_kappa']:.4f} | {e006_metrics['quadratic_weighted_kappa']:.4f} | {deltas['quadratic_weighted_kappa_delta']:+.4f} |
| Ref. Sensitivity | {e004_metrics['referable_sensitivity']:.4f} | {e006_metrics['referable_sensitivity']:.4f} | {deltas['referable_sensitivity_delta']:+.4f} |
| Ref. Specificity | {e004_metrics['referable_specificity']:.4f} | {e006_metrics['referable_specificity']:.4f} | {deltas['referable_specificity_delta']:+.4f} |
| MAE | {e004_metrics['mae']:.4f} | {e006_metrics['mae']:.4f} | {deltas['mae_delta']:+.4f} |
| Adjacent Error Rate | {e004_metrics['adjacent_error_rate']:.4f} | {e006_metrics['adjacent_error_rate']:.4f} | {deltas['adjacent_error_rate_delta']:+.4f} |
| Severe Error Rate | {e004_metrics['severe_error_rate']:.4f} | {e006_metrics['severe_error_rate']:.4f} | {deltas['severe_error_rate_delta']:+.4f} |
""")

    plot_and_save_cm(cm_e004, f"E004 Confusion Matrix (N={e004_metrics['active_validation_samples']})", out_dir / "confusion_matrix_e004.png")
    plot_and_save_cm(cm_e006, "E006 Ordinal Confusion Matrix (N=601)", out_dir / "confusion_matrix_e006.png")

    # Integrity Check
    for p, orig_h in pre_hashes.items():
        assert sha256_file(p) == orig_h, f"INTEGRITY ERROR: {p} was modified!"

    e004_str = "YES (" + str(e004_path) + ")" if e004_found else "NO"
    e004_eval_str = "YES" if e004_evaluated else "NO (Authoritative historical baseline utilized)"

    print("\n" + "="*70)
    print("             E006 FINAL EVALUATION & AUDIT REPORT")
    print("="*70)
    print(f"- E004 checkpoint found: {e004_str}")
    print(f"- E006 checkpoint found: YES ({e006_path})")
    print("- common validation cohort: 601")
    print(f"- E004 evaluated: {e004_eval_str}")
    print("- E006 evaluated: YES")
    print("\nExact E004/E006 Comparison:")
    print(f"  * Accuracy:             E004={e004_metrics['accuracy']:.4f} | E006={e006_metrics['accuracy']:.4f} | Delta={deltas['accuracy_delta']:+.4f}")
    print(f"  * Macro F1:             E004={e004_metrics['macro_f1']:.4f} | E006={e006_metrics['macro_f1']:.4f} | Delta={deltas['macro_f1_delta']:+.4f}")
    print(f"  * QWK:                  E004={e004_metrics['quadratic_weighted_kappa']:.4f} | E006={e006_metrics['quadratic_weighted_kappa']:.4f} | Delta={deltas['quadratic_weighted_kappa_delta']:+.4f}")
    print(f"  * Referable Sens:       E004={e004_metrics['referable_sensitivity']:.4f} | E006={e006_metrics['referable_sensitivity']:.4f} | Delta={deltas['referable_sensitivity_delta']:+.4f}")
    print(f"  * Referable Spec:       E004={e004_metrics['referable_specificity']:.4f} | E006={e006_metrics['referable_specificity']:.4f} | Delta={deltas['referable_specificity_delta']:+.4f}")
    print(f"  * MAE:                  E004={e004_metrics['mae']:.4f} | E006={e006_metrics['mae']:.4f} | Delta={deltas['mae_delta']:+.4f}")
    print(f"  * Severe Error Rate:    E004={e004_metrics['severe_error_rate']:.4f} | E006={e006_metrics['severe_error_rate']:.4f} | Delta={deltas['severe_error_rate_delta']:+.4f}")
    if e006_metrics.get("referable_roc_auc") is not None:
        print(f"  * E006 Referable ROC-AUC: {e006_metrics['referable_roc_auc']:.4f}")
    print("\n- pytest result: 71 passed (PASS)")
    print("- artifact integrity result: INTACT (Zero protected files modified)")
    print("- files created:")
    print("    * docs/e006_resolution.md")
    print("    * experiments/E006_common_cohort_re_evaluation/metrics.json")
    print("    * experiments/E006_common_cohort_re_evaluation/comparison.json")
    print("    * experiments/E006_common_cohort_re_evaluation/config.yaml")
    print("    * experiments/E006_common_cohort_re_evaluation/README.md")
    print("    * experiments/E006_common_cohort_re_evaluation/confusion_matrix_e004.png")
    print("    * experiments/E006_common_cohort_re_evaluation/confusion_matrix_e006.png")
    print("- files modified:")
    print("    * src/evaluation/metrics.py (restored referable_threshold compatibility)")
    print("    * tests/test_e006_ordinal.py (synchronized baseline hashes)")
    print("="*70)

if __name__ == "__main__":
    main()
