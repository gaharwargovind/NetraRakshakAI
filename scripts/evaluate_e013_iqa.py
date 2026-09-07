import os, sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.data.quality import FundusIQAExtractor
from src.inference.quality_gate import FundusQualityGate, QualityGateConfig, GateStatus
from src.models.efficientnet import DREfficientNet
from src.data.transforms import E001BaselineTransform
from src.robustness.benchmark import PERTURBATION_SUITE, apply_perturbation

EXPECTED_E007_SHA = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def compute_classification_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0, "qwk": 0.0, "referable_sensitivity": 0.0, "referable_specificity": 0.0}
    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic")) if len(np.unique(y_true)) > 1 else 0.0
    
    ref_t = (y_true >= 2).astype(int)
    ref_p = (y_pred >= 2).astype(int)
    tp = int(np.sum((ref_t == 1) & (ref_p == 1)))
    fn = int(np.sum((ref_t == 1) & (ref_p == 0)))
    tn = int(np.sum((ref_t == 0) & (ref_p == 0)))
    fp = int(np.sum((ref_t == 0) & (ref_p == 1)))
    
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "qwk": round(qwk, 4),
        "referable_sensitivity": round(sens, 4),
        "referable_specificity": round(spec, 4)
    }

def main():
    print("=" * 70)
    print("      EXPERIMENT E013: IMAGE QUALITY ASSESSMENT (IQA) BENCHMARK")
    print("=" * 70)

    ckpt_path = ROOT_DIR / "models/checkpoints/E007_best_model.pt"
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    test_csv = ROOT_DIR / "data/processed/aptos/test.csv"
    assert ckpt_path.is_file() and val_csv.is_file() and test_csv.is_file()

    ckpt_sha = sha256_file(ckpt_path)
    val_sha = sha256_file(val_csv)
    test_sha = sha256_file(test_csv)
    assert ckpt_sha == EXPECTED_E007_SHA
    val_df = pd.read_csv(val_csv)
    assert len(val_df) == 601
    assert sum(1 for _ in open(test_csv)) - 1 == 602

    print(f"[1] Verified Invariants: E007 Checkpoint Frozen, Val N=601, Test N=602 (LOCKED)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[2] Device: {device} | Loading E007 Model (Frozen Inference Mode)")
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    raw_images_dir = ROOT_DIR / "data/raw/aptos/train_images"
    transform = E001BaselineTransform(target_size=(512, 512))
    extractor = FundusIQAExtractor()
    gate = FundusQualityGate()

    print(f"\n[3] Evaluating Clean APTOS Validation Cohort (N=601)...")
    clean_records = []
    clean_pil_cache = []
    clean_targets = []
    clean_preds = []

    with torch.no_grad():
        for idx, row in val_df.iterrows():
            id_code = str(row["id_code"])
            diag = int(row["diagnosis"])
            img_path = raw_images_dir / f"{id_code}.png"
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
            clean_pil_cache.append(img_rgb)
            clean_targets.append(diag)

            tensor = transform(img_rgb).unsqueeze(0).to(device)
            logit = model(tensor).cpu().numpy()[0]
            pred = int(np.argmax(logit))
            clean_preds.append(pred)

            img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            eval_res = gate.evaluate(img_bgr)
            m = eval_res.metrics

            clean_records.append({
                "image_id": id_code,
                "true_grade": diag,
                "pred_grade": pred,
                "iqa_status": eval_res.status.value,
                "rejection_reasons": "; ".join(eval_res.rejection_reasons),
                "fov_coverage": m.fov_coverage,
                "laplacian_variance": m.laplacian_variance,
                "edge_density": m.edge_density,
                "mean_intensity": m.mean_intensity,
                "dark_fraction": m.dark_fraction,
                "bright_fraction": m.bright_fraction,
                "illumination_uniformity": m.illumination_uniformity,
                "intensity_std": m.intensity_std,
                "percentile_spread_90": m.percentile_spread_90,
                "noise_mad": m.noise_mad
            })

    clean_df = pd.DataFrame(clean_records)
    clean_targets = np.array(clean_targets)
    clean_preds = np.array(clean_preds)

    passed_mask = (clean_df["iqa_status"] == "PASS").values
    failed_mask = ~passed_mask

    clean_total = len(clean_df)
    clean_passed_count = int(np.sum(passed_mask))
    clean_failed_count = int(np.sum(failed_mask))
    false_rejection_rate = clean_failed_count / clean_total

    print(f"    Clean Validation Total:     {clean_total}")
    print(f"    Clean Images PASSED:        {clean_passed_count} ({clean_passed_count/clean_total*100:.2f}%)")
    print(f"    Clean Images REJECTED:      {clean_failed_count} ({false_rejection_rate*100:.2f}%)")

    metrics_all = compute_classification_metrics(clean_targets, clean_preds)
    metrics_pass = compute_classification_metrics(clean_targets[passed_mask], clean_preds[passed_mask])
    metrics_fail = compute_classification_metrics(clean_targets[failed_mask], clean_preds[failed_mask])

    print("\n[4] Downstream E007 Performance Breakdown:")
    print(f"    Unfiltered (N={clean_total}):      Acc={metrics_all['accuracy']:.4f} | F1={metrics_all['macro_f1']:.4f} | QWK={metrics_all['qwk']:.4f} | Sens={metrics_all['referable_sensitivity']:.4f}")
    print(f"    IQA-PASS (N={clean_passed_count}):        Acc={metrics_pass['accuracy']:.4f} | F1={metrics_pass['macro_f1']:.4f} | QWK={metrics_pass['qwk']:.4f} | Sens={metrics_pass['referable_sensitivity']:.4f}")
    print(f"    IQA-FAIL Diagnostic (N={clean_failed_count}): Acc={metrics_fail['accuracy']:.4f} | F1={metrics_fail['macro_f1']:.4f} | QWK={metrics_fail['qwk']:.4f} | Sens={metrics_fail['referable_sensitivity']:.4f}")

    distribution_summary = {}
    metric_cols = [
        "fov_coverage", "laplacian_variance", "edge_density", "mean_intensity",
        "dark_fraction", "bright_fraction", "illumination_uniformity",
        "intensity_std", "percentile_spread_90", "noise_mad"
    ]
    for c in metric_cols:
        vals = clean_df[c].values
        distribution_summary[c] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "p01": round(float(np.percentile(vals, 1)), 4),
            "p05": round(float(np.percentile(vals, 5)), 4),
            "median": round(float(np.median(vals)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
            "p99": round(float(np.percentile(vals, 99)), 4),
        }

    print("\n[5] Stress-Testing IQA Gate Across E011 Perturbation Families...")
    pert_results = {}
    
    for family, conditions in PERTURBATION_SUITE.items():
        pert_results[family] = []
        for cond in conditions:
            sev = cond["severity"]
            desc = cond["desc"]
            params = cond["params"]
            rejections = 0

            for i, img_rgb in enumerate(clean_pil_cache):
                pert_img = apply_perturbation(family, img_rgb, params, seed=42000 + i)
                pert_bgr = cv2.cvtColor(np.array(pert_img), cv2.COLOR_RGB2BGR)
                res = gate.evaluate(pert_bgr)
                if res.status == GateStatus.FAIL:
                    rejections += 1

            rej_rate = rejections / clean_total
            pert_results[family].append({
                "severity": sev,
                "description": desc,
                "rejected_count": rejections,
                "rejection_rate": round(rej_rate, 4)
            })
            print(f"    [{family:<17}] Sev {sev} ({desc:<30}) -> Rejection Rate: {rej_rate*100:5.2f}%")

    exp_dir = ROOT_DIR / "experiments/E013_iqa"
    fig_dir = exp_dir / "figures"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for family, rows in pert_results.items():
        sevs = [r["severity"] for r in rows]
        rates = [r["rejection_rate"] * 100 for r in rows]
        ax.plot(sevs, rates, marker="o", linewidth=2.0, label=family.replace("_", " ").title())
    ax.axhline(false_rejection_rate * 100, linestyle="--", color="black", alpha=0.7, label=f"Clean Baseline ({false_rejection_rate*100:.2f}%)")
    ax.set_title("IQA Gate Rejection Rate vs. Perturbation Severity", fontweight="bold", fontsize=11, pad=10)
    ax.set_xlabel("Severity Level", fontsize=10)
    ax.set_ylabel("Rejection Rate (%)", fontsize=10)
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "iqa_rejection_curves.png", dpi=250)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(clean_df["laplacian_variance"], bins=30, color="royalblue", edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(4.0, color="red", linestyle="--", label="Calibrated Threshold (4.0)")
    axes[0, 0].set_title("Laplacian Variance (Focus)", fontweight="bold")
    axes[0, 0].legend()

    axes[0, 1].hist(clean_df["percentile_spread_90"], bins=30, color="teal", edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(20.0, color="red", linestyle="--", label="Calibrated Threshold (20.0)")
    axes[0, 1].set_title("90% Percentile Spread (Contrast)", fontweight="bold")
    axes[0, 1].legend()

    axes[1, 0].hist(clean_df["dark_fraction"], bins=30, color="darkorange", edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(0.15, color="red", linestyle="--", label="Calibrated Threshold (0.15)")
    axes[1, 0].set_title("Dark Pixel Fraction", fontweight="bold")
    axes[1, 0].legend()

    axes[1, 1].hist(clean_df["noise_mad"], bins=30, color="crimson", edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(4.0, color="red", linestyle="--", label="Calibrated Threshold (4.0)")
    axes[1, 1].set_title("Noise Proxy (Donoho MAD)", fontweight="bold")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "clean_distributions.png", dpi=250)
    plt.close()

    clean_df.to_csv(exp_dir / "iqa_results.csv", index=False)

    metrics_payload = {
        "experiment_id": "E013",
        "name": "Image Quality Assessment (IQA) Benchmark",
        "clean_validation": {
            "total_evaluated": clean_total,
            "passed_count": clean_passed_count,
            "rejected_count": clean_failed_count,
            "false_rejection_rate": round(false_rejection_rate, 4)
        },
        "downstream_e007_impact": {
            "unfiltered": metrics_all,
            "iqa_pass": metrics_pass,
            "iqa_fail_diagnostic": metrics_fail
        },
        "clean_distributions": distribution_summary,
        "perturbation_rejection_rates": pert_results,
        "governance": {
            "e007_checkpoint_sha256": ckpt_sha,
            "aptos_val_sha256": val_sha,
            "aptos_test_sha256": test_sha,
            "scientific_mandate": "IQA is an upstream safety mechanism intended to identify potentially ungradable images; clinical validation is still required."
        }
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    readme_content = f"""# Experiment E013: Image Quality Assessment Benchmark

## Executive Summary
- **Status:** COMPLETE
- **Upstream Role:** Deterministic pre-inference quality triage.
- **Clean False Rejection Rate:** `{false_rejection_rate*100:.2f}%` ({clean_failed_count}/{clean_total})
- **Downstream E007 Sensitivity (IQA-PASS):** `{metrics_pass['referable_sensitivity']*100:.2f}%` (Unfiltered: `{metrics_all['referable_sensitivity']*100:.2f}%`)
- **Downstream E007 QWK (IQA-PASS):** `{metrics_pass['qwk']:.4f}` (Unfiltered: `{metrics_all['qwk']:.4f}`)

## Scientific Safety Mandate
"IQA is an upstream safety mechanism intended to identify potentially ungradable images; clinical validation is still required."
"""
    (exp_dir / "README.md").write_text(readme_content)

    assert sha256_file(ckpt_path) == EXPECTED_E007_SHA
    assert sha256_file(val_csv) == val_sha
    assert sha256_file(test_csv) == test_sha

    print("\n" + "=" * 70)
    print("      EXPERIMENT E013 BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"- Clean Rejection Rate:    {false_rejection_rate*100:.2f}% ({clean_failed_count}/{clean_total})")
    print(f"- E007 QWK (Passed):       {metrics_pass['qwk']:.4f} (Unfiltered: {metrics_all['qwk']:.4f})")
    print(f"- E007 Sens (Passed):      {metrics_pass['referable_sensitivity']:.4f} (Unfiltered: {metrics_all['referable_sensitivity']:.4f})")
    print(f"- E007 Sens (Failed):      {metrics_fail['referable_sensitivity']:.4f} (Diagnostic evidence)")
    print(f"- Checkpoint Integrity:    PASSED ({ckpt_sha})")
    print("=" * 70)

if __name__ == "__main__":
    main()
