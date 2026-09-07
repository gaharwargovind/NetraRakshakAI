import os, sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    precision_recall_fscore_support, confusion_matrix, roc_auc_score
)

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.models.efficientnet import DREfficientNet
from src.data.transforms import E001BaselineTransform
from src.evaluation.calibration import (
    compute_multiclass_ece, compute_multiclass_metrics, compute_referable_calibration
)

FROZEN_T = 0.7785
EXPECTED_E007_SHA = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("      EXPERIMENT E012: FROZEN EXTERNAL VALIDATION (IDRiD)")
    print("=" * 70)

    ckpt_path = ROOT_DIR / "models/checkpoints/E007_best_model.pt"
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    test_csv = ROOT_DIR / "data/processed/aptos/test.csv"
    assert ckpt_path.is_file() and val_csv.is_file() and test_csv.is_file()

    ckpt_sha = sha256_file(ckpt_path)
    val_sha = sha256_file(val_csv)
    test_sha = sha256_file(test_csv)
    assert ckpt_sha == EXPECTED_E007_SHA
    assert len(pd.read_csv(val_csv)) == 601
    assert sum(1 for _ in open(test_csv)) - 1 == 602

    print(f"[1] Verified Invariants: E007 SHA verified, Validation N=601, Test N=602 (LOCKED)")

    raw_idrid = ROOT_DIR / "data/raw/idrid"
    csv_paths = [
        p for p in raw_idrid.rglob("*.csv")
        if not p.name.startswith(".") and "mapping" not in p.name and "duplicate" not in p.name
    ]
    label_df = pd.read_csv(csv_paths[0])
    img_col = next(c for c in label_df.columns if "id" in c.lower() or "image" in c.lower())
    grade_col = next(c for c in label_df.columns if "diag" in c.lower() or "grade" in c.lower() or "retino" in c.lower())

    gt_map = {}
    for _, row in label_df.iterrows():
        cid = str(row[img_col]).replace(".jpg", "").replace(".png", "").strip().lower()
        try:
            gt_map[cid] = int(row[grade_col])
        except:
            continue

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_paths = [p for p in raw_idrid.rglob("*") if p.suffix.lower() in valid_exts and not p.name.startswith(".")]
    matched = []
    for p in image_paths:
        stem = p.stem.replace(".jpg", "").replace(".png", "").strip().lower()
        if stem in gt_map:
            matched.append((p.stem, p, gt_map[stem]))

    assert len(matched) == 455, f"Expected 455 matched cases, got {len(matched)}"
    print(f"[2] External Cohort Loaded: {len(matched)} IDRiD records verified.")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[3] Device: {device} | Loading E007 (Frozen Eval Mode)")
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    transform = E001BaselineTransform(target_size=(512, 512))
    raw_logits, y_true_list, image_ids = [], [], []

    print("[4] Executing deterministic forward pass on 455 IDRiD images...")
    with torch.no_grad():
        for img_id, img_path, true_grade in matched:
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            raw_logits.append(model(tensor).cpu().numpy()[0])
            y_true_list.append(true_grade)
            image_ids.append(img_id)

    logits = np.array(raw_logits, dtype=np.float32)
    y_true = np.array(y_true_list, dtype=np.int64)

    uncal_probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    cal_logits = logits / FROZEN_T
    cal_probs = np.exp(cal_logits) / np.sum(np.exp(cal_logits), axis=1, keepdims=True)

    y_pred = np.argmax(uncal_probs, axis=1)
    y_pred_cal = np.argmax(cal_probs, axis=1)
    assert np.array_equal(y_pred, y_pred_cal)

    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    adj_err = float(np.mean(np.abs(y_true - y_pred) == 1))
    sev_err = float(np.mean(np.abs(y_true - y_pred) >= 2))

    ref_t = (y_true >= 2).astype(int)
    ref_p = (y_pred >= 2).astype(int)
    tp = int(np.sum((ref_t == 1) & (ref_p == 1)))
    fn = int(np.sum((ref_t == 1) & (ref_p == 0)))
    tn = int(np.sum((ref_t == 0) & (ref_p == 0)))
    fp = int(np.sum((ref_t == 0) & (ref_p == 1)))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    pr, rc, f1, sp = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3,4], zero_division=0)
    per_class = {str(i): {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4), "support": int(sp[i])} for i in range(5)}
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])

    uncal_ece, _, _ = compute_multiclass_ece(uncal_probs, y_true, n_bins=10)
    cal_ece, _, _ = compute_multiclass_ece(cal_probs, y_true, n_bins=10)
    uncal_brier, uncal_nll = compute_multiclass_metrics(uncal_probs, y_true)
    cal_brier, cal_nll = compute_multiclass_metrics(cal_probs, y_true)
    uncal_ref_ece, uncal_ref_brier, uncal_ref_nll, _ = compute_referable_calibration(uncal_probs, y_true, n_bins=10)
    cal_ref_ece, cal_ref_brier, cal_ref_nll, _ = compute_referable_calibration(cal_probs, y_true, n_bins=10)

    exp_dir = ROOT_DIR / "experiments/E012_external_validation"
    exp_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame([{
        "image_id": image_ids[i], "true_grade": int(y_true[i]), "predicted_grade": int(y_pred[i]),
        "confidence_cal": round(float(np.max(cal_probs[i])), 4), "p_ref_cal": round(float(np.sum(cal_probs[i, 2:])), 4),
        "referable_true": int(ref_t[i]), "referable_pred": int(ref_p[i]), "correct": int(y_true[i] == y_pred[i]),
        "grade_error": int(abs(y_true[i] - y_pred[i]))
    } for i in range(len(image_ids))])
    pred_df.to_csv(exp_dir / "external_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_title("IDRiD External Validation Confusion Matrix (N=455)", fontweight="bold")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"G{i}" for i in range(5)]); ax.set_yticklabels([f"G{i}" for i in range(5)])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(exp_dir / "confusion_matrix.png", dpi=300); plt.close()

    metrics_payload = {
        "experiment_id": "E012", "dataset": "IDRiD", "cohort_size": 455,
        "primary_metrics": {"macro_f1": round(mf1, 4), "referable_sensitivity": round(sens, 4), "referable_specificity": round(spec, 4), "qwk": round(qwk, 4)},
        "secondary_metrics": {"accuracy": round(acc, 4), "weighted_f1": round(wf1, 4), "mae": round(mae, 4), "adjacent_grade_error": round(adj_err, 4), "severe_grade_error": round(sev_err, 4)},
        "internal_vs_external_deltas": {
            "delta_macro_f1": round(mf1 - 0.7123, 4), "delta_qwk": round(qwk - 0.9156, 4),
            "delta_sensitivity": round(sens - 0.9444, 4), "delta_specificity": round(spec - 0.9427, 4), "delta_accuracy": round(acc - 0.8353, 4)
        },
        "per_class": per_class, "confusion_matrix": cm.tolist(),
        "calibration": {"raw_ece": uncal_ece, "calibrated_ece": cal_ece, "raw_ref_ece": uncal_ref_ece, "calibrated_ref_ece": cal_ref_ece}
    }
    with open(exp_dir / "metrics.json", "w") as f: json.dump(metrics_payload, f, indent=2)

    status_str = "PASSED" if sens >= 0.90 else "FAILED"
    readme_content = f"# Experiment E012: External Validation (IDRiD)\n\n- Safety Gate: {status_str} (Sensitivity: {sens:.4f})\n- Macro F1: {mf1:.4f}\n- QWK: {qwk:.4f}\n- Accuracy: {acc:.4f}\n- Referable Specificity: {spec:.4f}\n"
    with open(exp_dir / "README.md", "w") as f: f.write(readme_content)

    assert sha256_file(ckpt_path) == EXPECTED_E007_SHA
    assert sha256_file(val_csv) == val_sha
    assert sha256_file(test_csv) == test_sha

    print("\n" + "=" * 70)
    print("      EXPERIMENT E012 EXTERNAL VALIDATION COMPLETE")
    print("=" * 70)
    print(f"- Cohort Size:             IDRiD (N={len(matched)})")
    print(f"- Macro F1:                {mf1:.4f} (APTOS Val: 0.7123 | Delta: {mf1 - 0.7123:+.4f})")
    print(f"- QWK:                     {qwk:.4f} (APTOS Val: 0.9156 | Delta: {qwk - 0.9156:+.4f})")
    print(f"- Referable Sensitivity:   {sens:.4f} ({status_str})")
    print(f"- Referable Specificity:   {spec:.4f} (APTOS Val: 0.9427 | Delta: {spec - 0.9427:+.4f})")
    print(f"- Accuracy:                {acc:.4f} (APTOS Val: 0.8353 | Delta: {acc - 0.8353:+.4f})")
    print(f"- MAE:                     {mae:.4f}")
    print(f"- Multiclass ECE (Frozen): {cal_ece:.4f} (Raw: {uncal_ece:.4f})")
    print(f"- Referable ECE (Frozen):  {cal_ref_ece:.4f} (Raw: {uncal_ref_ece:.4f})")
    print(f"- Safety Gate Status:      {status_str}")
    print("=" * 70)

if __name__ == "__main__":
    main()
