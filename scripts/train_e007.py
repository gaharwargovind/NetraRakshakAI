import json, time, random, hashlib, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.models.efficientnet import DREfficientNet
from src.losses.focal_loss import MulticlassFocalLoss
from src.data.loaders import create_e004_data_loaders

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, device):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for imgs, tgts, *_ in loader:
            p = torch.softmax(model(imgs.to(device)), dim=-1)
            preds.extend(torch.argmax(p, dim=-1).cpu().numpy().tolist())
            targets.extend(tgts.numpy().tolist())
            probs.append(p.cpu().numpy())
    y_t, y_p = np.array(targets, dtype=np.int64), np.array(preds, dtype=np.int64)
    probs_arr = np.vstack(probs)
    acc = float(accuracy_score(y_t, y_p))
    mf1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    wf1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
    qwk = float(cohen_kappa_score(y_t, y_p, weights="quadratic"))
    ref_t, ref_p = (y_t >= 2).astype(int), (y_p >= 2).astype(int)
    tp = int(np.sum((ref_t == 1) & (ref_p == 1)))
    fn = int(np.sum((ref_t == 1) & (ref_p == 0)))
    tn = int(np.sum((ref_t == 0) & (ref_p == 0)))
    fp = int(np.sum((ref_t == 0) & (ref_p == 1)))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    pr, rc, f1, sp = precision_recall_fscore_support(y_t, y_p, labels=[0,1,2,3,4], zero_division=0)
    per_class = {str(i): {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4), "f1": round(float(f1[i]), 4), "support": int(sp[i])} for i in range(5)}
    cm = confusion_matrix(y_t, y_p, labels=[0,1,2,3,4]).tolist()
    try:
        macro_auc = float(roc_auc_score(np.eye(5)[y_t], probs_arr, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = None
    return {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "weighted_f1": round(wf1, 4),
            "quadratic_weighted_kappa": round(qwk, 4), "referable_sensitivity": round(sens, 4),
            "referable_specificity": round(spec, 4), "macro_roc_auc": round(macro_auc, 4) if macro_auc else None,
            "per_class": per_class, "confusion_matrix": cm}

def save_cm_plot(cm, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_title("E007 Confusion Matrix (N=601)", fontweight="bold")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([f"G{i}" for i in range(5)]); ax.set_yticklabels([f"G{i}" for i in range(5)])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

def main():
    set_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    assert len(pd.read_csv(val_csv)) == 601, "Validation cohort must be 601"
    base_cfg = {
        "compute": {"device": "mps" if torch.backends.mps.is_available() else "cpu", "num_workers": 2},
        "data": {"image_size": [512, 512]}
    }
    exp_cfg = {
        "training": {"batch_size": 16},
        "augmentation": {
            "horizontal_flip": {"probability": 0.5},
            "rotation": {"probability": 0.7, "max_degrees": 10.0},
            "affine": {"probability": 0.5, "scale_range": [0.96, 1.04], "translate_pct": 0.02},
            "color_jitter": {
                "probability": 0.5,
                "brightness_factor_range": [0.95, 1.05],
                "contrast_factor_range": [0.95, 1.05]
            }
        }
    }
    train_loader, val_loader = create_e004_data_loaders(base_cfg, exp_cfg, project_root=ROOT_DIR)
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=True, dropout_rate=0.2).to(device)
    criterion = MulticlassFocalLoss(gamma=2.0, reduction="mean")
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    exp_dir = ROOT_DIR / "experiments/E007_focal_loss"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT_DIR / "models/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, best_res, patience_cnt = -1.0, -1, {}, 0
    history, start_t = [], time.time()
    print("Beginning E007 training...")
    for ep in range(1, 31):
        t0 = time.time(); model.train(); loss_sum, steps = 0.0, 0
        for imgs, tgts, *_ in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs.to(device)), tgts.to(device))
            loss.backward(); optimizer.step()
            loss_sum += loss.item(); steps += 1
        scheduler.step(); loss_avg = loss_sum / max(1, steps)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": round(loss_avg, 4), "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"], "val_qwk": val["quadratic_weighted_kappa"],
                        "val_sensitivity": val["referable_sensitivity"], "val_specificity": val["referable_specificity"],
                        "lr": optimizer.param_groups[0]["lr"], "duration": round(time.time() - t0, 1)})
        print(f"Epoch {ep:02d}/30 [{time.time()-t0:.1f}s] Loss: {loss_avg:.4f} | F1: {val['macro_f1']:.4f} | QWK: {val['quadratic_weighted_kappa']:.4f} | Sens: {val['referable_sensitivity']:.4f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, best_res, patience_cnt = val["macro_f1"], ep, val, 0
            ckpt = {"epoch": ep, "model_state_dict": model.state_dict(), "best_metric": best_f1,
                    "config": {"experiment": "E007", "loss": "multiclass_focal_loss", "gamma": 2.0, "alpha": 1.0}}
            torch.save(ckpt, ckpt_dir / "E007_best_model.pt")
            torch.save(ckpt, exp_dir / "model_checkpoint.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= 7:
                print(f"Early stopping at epoch {ep}")
                break
    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)
    payload = {"experiment_id": "E007", "best_epoch": best_epoch, "duration": round(time.time() - start_t, 1),
               "validation_count": 601, "factor": "Weighted CE -> Multiclass Focal Loss (gamma=2.0)", "metrics": best_res}
    with open(exp_dir / "metrics.json", "w") as f: json.dump(payload, f, indent=2)
    cfg_src = ROOT_DIR / "configs/experiments/focal_loss.yaml"
    if cfg_src.is_file(): shutil.copy(cfg_src, exp_dir / "config.yaml")
    cm_arr = np.array(best_res["confusion_matrix"])
    save_cm_plot(cm_arr, exp_dir / "confusion_matrix.png")
    sens_pass = best_res["referable_sensitivity"] >= 0.90
    readme_text = f"# Experiment E007: Multiclass Focal Loss\n\n- Status: {'PASSED_SENSITIVITY' if sens_pass else 'FAILED_SENSITIVITY'}\n- Best Epoch: {best_epoch}\n- Macro F1: {best_res['macro_f1']:.4f}\n- Accuracy: {best_res['accuracy']:.4f}\n- QWK: {best_res['quadratic_weighted_kappa']:.4f}\n- Referable Sens: {best_res['referable_sensitivity']:.4f}\n- Referable Spec: {best_res['referable_specificity']:.4f}\n\nConfusion Matrix:\n{cm_arr}\n"
    with open(exp_dir / "README.md", "w") as f: f.write(readme_text)
    reg_path = ROOT_DIR / "experiments/experiment_registry.csv"
    if reg_path.is_file():
        reg_df = pd.read_csv(reg_path)
        row = {"experiment_id": "E007", "experiment_name": "Multiclass Focal Loss", "backbone": "efficientnet_b0",
               "loss": "multiclass_focal_loss", "macro_f1": best_res["macro_f1"], "accuracy": best_res["accuracy"],
               "qwk": best_res["quadratic_weighted_kappa"], "referable_sensitivity": best_res["referable_sensitivity"],
               "referable_specificity": best_res["referable_specificity"], "validation_samples": 601,
               "status": "VALIDATED" if sens_pass else "FAILED_SENSITIVITY", "notes": f"Single factor vs E004. Best epoch {best_epoch}."}
        for c in reg_df.columns:
            if c not in row: row[c] = ""
        pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True).to_csv(reg_path, index=False)
    print("=" * 70)
    print(f"E007 COMPLETE | Best Epoch: {best_epoch} | F1: {best_res['macro_f1']:.4f} | QWK: {best_res['quadratic_weighted_kappa']:.4f} | Sens: {best_res['referable_sensitivity']:.4f} | Spec: {best_res['referable_specificity']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
