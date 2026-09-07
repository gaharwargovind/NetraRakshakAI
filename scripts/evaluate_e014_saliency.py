import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.models.efficientnet import DREfficientNet
from src.data.transforms import E001BaselineTransform
from src.data.quality import FundusIQAExtractor

EXPECTED_E007_SHA = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, output.detach()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

def compute_probs(logits):
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / np.sum(exp_l)
    p_ref = float(np.sum(probs[2:]))
    pred_grade = int(np.argmax(logits))
    return pred_grade, p_ref, probs

def main():
    print("=" * 70)
    print("      EXPERIMENT E014: COUNTERFACTUAL & VISUAL SALIENCY AUDIT")
    print("=" * 70)

    ckpt_path = ROOT_DIR / "models/checkpoints/E007_best_model.pt"
    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    idrid_csv = ROOT_DIR / "data/processed/idrid/idrid_manifest.csv"
    aptos_img_dir = ROOT_DIR / "data/raw/aptos/train_images"
    idrid_img_dir = ROOT_DIR / "data/raw/idrid/images"
    
    # 1. Verify Invariants
    ckpt_sha = sha256_file(ckpt_path)
    print(f"[1] Verified Checkpoint SHA: {ckpt_sha}")
    assert ckpt_sha == EXPECTED_E007_SHA, "E007 checkpoint SHA modified!"

    val_df = pd.read_csv(val_csv)
    idrid_df = pd.read_csv(idrid_csv)
    assert len(val_df) == 601
    assert len(idrid_df) == 455
    print(f"[2] Verified Manifests: APTOS Val N={len(val_df)}, IDRiD Manifest N={len(idrid_df)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[3] Device: {device} | Loading E007 Architecture")
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_layer = model.model.features[-1]
    cam_engine = GradCAM(model, target_layer)
    transform = E001BaselineTransform(target_size=(512, 512))
    extractor = FundusIQAExtractor()

    # Pre-evaluate validation and external predictions to select deterministic cases
    print("\n[4] Running Deterministic Case Selection...")
    aptos_preds = []
    with torch.no_grad():
        for idx, row in val_df.iterrows():
            id_c = str(row["id_code"])
            true_g = int(row["diagnosis"])
            p = aptos_img_dir / f"{id_c}.png"
            with Image.open(p) as img_f:
                rgb = img_f.convert("RGB")
            t = transform(rgb).unsqueeze(0).to(device)
            l = model(t).cpu().numpy()[0]
            pred_g, p_ref, probs = compute_probs(l)
            margin = float(np.sort(probs)[-1] - np.sort(probs)[-2])
            max_p = float(np.max(probs))
            aptos_preds.append({
                "id_code": id_c, "true_grade": true_g, "pred_grade": pred_g,
                "p_ref": p_ref, "max_prob": max_p, "margin": margin,
                "correct": (true_g == pred_g)
            })
    aptos_eval_df = pd.DataFrame(aptos_preds)

    idrid_preds = []
    with torch.no_grad():
        for idx, row in idrid_df.iterrows():
            id_c = str(row["image_id"])
            true_g = int(row["dr_grade"])
            p = idrid_img_dir / f"{id_c}.jpg"
            with Image.open(p) as img_f:
                rgb = img_f.convert("RGB")
            t = transform(rgb).unsqueeze(0).to(device)
            l = model(t).cpu().numpy()[0]
            pred_g, p_ref, probs = compute_probs(l)
            idrid_preds.append({
                "image_id": id_c, "true_grade": true_g, "pred_grade": pred_g,
                "p_ref": p_ref, "correct": (true_g == pred_g)
            })
    idrid_eval_df = pd.DataFrame(idrid_preds)

    # Deterministic Selection Rules
    cases_selected = []
    # A. Correct Grade 0
    c0 = aptos_eval_df[(aptos_eval_df["true_grade"] == 0) & aptos_eval_df["correct"]].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "A_correct_grade_0", "dataset": "APTOS", "id": c0["id_code"], "true": 0, "path": aptos_img_dir / f"{c0['id_code']}.png"})
    
    # B. Correct Grade 1
    c1 = aptos_eval_df[(aptos_eval_df["true_grade"] == 1) & aptos_eval_df["correct"]].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "B_correct_grade_1", "dataset": "APTOS", "id": c1["id_code"], "true": 1, "path": aptos_img_dir / f"{c1['id_code']}.png"})

    # C. Correct Grade 2
    c2 = aptos_eval_df[(aptos_eval_df["true_grade"] == 2) & aptos_eval_df["correct"]].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "C_correct_grade_2", "dataset": "APTOS", "id": c2["id_code"], "true": 2, "path": aptos_img_dir / f"{c2['id_code']}.png"})

    # D. Correct Grade 3
    c3 = aptos_eval_df[(aptos_eval_df["true_grade"] == 3) & aptos_eval_df["correct"]].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "D_correct_grade_3", "dataset": "APTOS", "id": c3["id_code"], "true": 3, "path": aptos_img_dir / f"{c3['id_code']}.png"})

    # E. Correct Grade 4
    c4 = aptos_eval_df[(aptos_eval_df["true_grade"] == 4) & aptos_eval_df["correct"]].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "E_correct_grade_4", "dataset": "APTOS", "id": c4["id_code"], "true": 4, "path": aptos_img_dir / f"{c4['id_code']}.png"})

    # F. False negative referable case (True >= 2, Pred < 2)
    fn_c = aptos_eval_df[(aptos_eval_df["true_grade"] >= 2) & (aptos_eval_df["pred_grade"] < 2)].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "F_false_negative_referable", "dataset": "APTOS", "id": fn_c["id_code"], "true": int(fn_c["true_grade"]), "path": aptos_img_dir / f"{fn_c['id_code']}.png"})

    # G. False positive referable case (True < 2, Pred >= 2)
    fp_c = aptos_eval_df[(aptos_eval_df["true_grade"] < 2) & (aptos_eval_df["pred_grade"] >= 2)].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "G_false_positive_referable", "dataset": "APTOS", "id": fp_c["id_code"], "true": int(fp_c["true_grade"]), "path": aptos_img_dir / f"{fp_c['id_code']}.png"})

    # H. High-confidence wrong prediction
    hc_w = aptos_eval_df[~aptos_eval_df["correct"]].sort_values("max_prob", ascending=False).iloc[0]
    cases_selected.append({"protocol": "H_high_confidence_wrong", "dataset": "APTOS", "id": hc_w["id_code"], "true": int(hc_w["true_grade"]), "path": aptos_img_dir / f"{hc_w['id_code']}.png"})

    # I. Low-confidence prediction (smallest margin)
    lc_p = aptos_eval_df.sort_values("margin", ascending=True).iloc[0]
    cases_selected.append({"protocol": "I_low_confidence", "dataset": "APTOS", "id": lc_p["id_code"], "true": int(lc_p["true_grade"]), "path": aptos_img_dir / f"{lc_p['id_code']}.png"})

    # J. External IDRiD domain-shift failure (True=2, Pred<2)
    idrid_fn = idrid_eval_df[(idrid_eval_df["true_grade"] == 2) & (idrid_eval_df["pred_grade"] < 2)].iloc[0]
    cases_selected.append({"protocol": "J_external_idrid_failure", "dataset": "IDRiD", "id": idrid_fn["image_id"], "true": 2, "path": idrid_img_dir / f"{idrid_fn['image_id']}.jpg"})

    print(f"[*] Deterministically selected {len(cases_selected)} audit cases across categories A-J.")

    out_dir = ROOT_DIR / "experiments/E014_saliency_audit"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_table = []

    print("\n[5] Executing Grad-CAM, Saliency Energy, and Counterfactual Ablation...")
    for item in cases_selected:
        prot = item["protocol"]
        cid = item["id"]
        true_g = item["true"]
        cpath = item["path"]

        with Image.open(cpath) as img_f:
            pil_rgb = img_f.convert("RGB")

        input_tensor = transform(pil_rgb).unsqueeze(0).to(device)
        cam, orig_logits = cam_engine.generate(input_tensor)
        orig_l = orig_logits.cpu().numpy()[0]
        orig_pred, orig_pref, orig_probs = compute_probs(orig_l)

        inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        unnorm = (input_tensor[0] * inv_std + inv_mean).clamp(0, 1)
        rgb_512 = (unnorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        bgr_512 = cv2.cvtColor(rgb_512, cv2.COLOR_RGB2BGR)

        mask_retina = extractor.extract_retinal_mask(bgr_512)
        total_cam_energy = float(np.sum(cam))
        fg_cam_energy = float(np.sum(cam[mask_retina > 0]))
        bg_cam_energy = float(np.sum(cam[mask_retina == 0]))
        leakage_ratio = (bg_cam_energy / total_cam_energy) if total_cam_energy > 0 else 0.0

        salient_mask = (cam >= 0.50)
        bg_mask = (cam < 0.50)

        # 1. Salient-region masked
        sal_masked_np = rgb_512.copy()
        sal_masked_np[salient_mask] = 0
        sal_pil = Image.fromarray(sal_masked_np)
        sal_tensor = transform(sal_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            sal_l = model(sal_tensor).cpu().numpy()[0]
        sal_pred, sal_pref, sal_probs = compute_probs(sal_l)

        # 2. Background-region masked
        bg_masked_np = rgb_512.copy()
        bg_masked_np[bg_mask] = 0
        bg_pil = Image.fromarray(bg_masked_np)
        bg_tensor = transform(bg_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            bg_l = model(bg_tensor).cpu().numpy()[0]
        bg_pred, bg_pref, bg_probs = compute_probs(bg_l)

        # Overlay Generation
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.45 * heatmap + 0.55 * rgb_512)

        # Plot Quad Panel
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        axes[0].imshow(rgb_512)
        axes[0].set_title(f"Original: {cid}\nTrue: G{true_g} | Pred: G{orig_pred} (P_ref={orig_pref:.2f})", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM (features[-1])\nRetinal Energy: {fg_cam_energy/total_cam_energy*100:.1f}%", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(sal_masked_np)
        axes[2].set_title(f"Salient Masked (CAM >= 0.5)\nPred: G{sal_pred} (P_ref={sal_pref:.2f} | Δ={sal_pref-orig_pref:+.2f})", fontsize=10)
        axes[2].axis("off")

        axes[3].imshow(bg_masked_np)
        axes[3].set_title(f"Background Masked\nPred: G{bg_pred} (P_ref={bg_pref:.2f} | Δ={bg_pref-orig_pref:+.2f})", fontsize=10)
        axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(fig_dir / f"{prot}_{cid}.png", dpi=200)
        plt.close()

        results_table.append({
            "protocol": prot,
            "dataset": item["dataset"],
            "image_id": cid,
            "true_grade": true_g,
            "orig_pred_grade": orig_pred,
            "orig_p_referable": round(orig_pref, 4),
            "salient_masked_pred": sal_pred,
            "salient_masked_p_ref": round(sal_pref, 4),
            "salient_delta_p_ref": round(sal_pref - orig_pref, 4),
            "bg_masked_pred": bg_pred,
            "bg_masked_p_ref": round(bg_pref, 4),
            "bg_delta_p_ref": round(bg_pref - orig_pref, 4),
            "retinal_energy_fraction": round(fg_cam_energy / total_cam_energy, 4),
            "non_retinal_leakage_fraction": round(leakage_ratio, 4)
        })
        print(f"  [{prot:<28}] Orig: G{orig_pred} (Pref={orig_pref:.2f}) -> SalMask: G{sal_pred} (Pref={sal_pref:.2f}) | Retinal Saliency: {fg_cam_energy/total_cam_energy*100:5.1f}%")

    cam_engine.remove_hooks()

    cases_df = pd.DataFrame(results_table)
    cases_df.to_csv(out_dir / "cases.csv", index=False)

    avg_leakage = float(cases_df["non_retinal_leakage_fraction"].mean())
    avg_sal_delta = float(cases_df["salient_delta_p_ref"].mean())
    avg_bg_delta = float(cases_df["bg_delta_p_ref"].mean())

    metrics_payload = {
        "experiment_id": "E014",
        "name": "Counterfactual & Visual Saliency Audit",
        "step_0_audit": {
            "idrid_lesion_masks_available": False,
            "status_statement": "Lesion-level quantitative validation unavailable for this cohort.",
            "iou_dice_computed": False
        },
        "target_layer": "model.model.features[-1]",
        "total_cases_analyzed": len(cases_df),
        "counterfactual_summary": {
            "mean_salient_ablation_delta_p_ref": round(avg_sal_delta, 4),
            "mean_background_ablation_delta_p_ref": round(avg_bg_delta, 4),
            "salient_ablation_grade_shift_rate": round(float(np.mean(cases_df["orig_pred_grade"] != cases_df["salient_masked_pred"])), 4)
        },
        "saliency_leakage_summary": {
            "mean_retinal_energy_fraction": round(float(cases_df["retinal_energy_fraction"].mean()), 4),
            "mean_non_retinal_leakage_fraction": round(avg_leakage, 4),
            "max_non_retinal_leakage_fraction": round(float(cases_df["non_retinal_leakage_fraction"].max()), 4)
        },
        "governance": {
            "e007_checkpoint_sha256": ckpt_sha,
            "aptos_test_accessed": False,
            "scientific_mandate": "Grad-CAM provides attribution evidence regarding image regions influencing the model output; clinical validation of gradability remains necessary."
        }
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    config_yaml = """experiment_id: E014_saliency_audit
phase: PHASE_E014_COUNTERFACTUAL_SALIENCY_AUDIT
base_model: E007
target_layer: model.model.features[-1]
governance:
  e007_weights: FROZEN
  aptos_test_split: UNTOUCHED
  lesion_overlap_metric: DISABLED_NO_GROUND_TRUTH_MASKS
counterfactual:
  threshold: 0.50
  ablation_modes:
    - salient_masked
    - background_masked
"""
    (out_dir / "config.yaml").write_text(config_yaml)

    readme_content = f"""# Experiment E014: Counterfactual & Visual Saliency Audit

## Summary
- **Status:** COMPLETE
- **Lesion Annotations:** Unavailable in raw IDRiD directory. Quantitative IoU/Dice suspended per Step 0.
- **Analyzed Cases:** {len(cases_df)} deterministic categories (Protocols A through J).
- **Mean Retinal Attribution Energy:** `{cases_df['retinal_energy_fraction'].mean()*100:.2f}%`
- **Mean Non-Retinal Saliency Leakage:** `{avg_leakage*100:.2f}%`
- **Salient Ablation Grade Shift Rate:** `{float(np.mean(cases_df['orig_pred_grade'] != cases_df['salient_masked_pred']))*100:.1f}%`

## Governance Mandate
Grad-CAM provides attribution evidence regarding image regions influencing the model output; clinical validation of gradability remains necessary.
"""
    (out_dir / "README.md").write_text(readme_content)

    print("\n" + "=" * 70)
    print("      EXPERIMENT E014 SALIENCY AUDIT COMPLETE")
    print("=" * 70)
    print(f"- Cases Audited:           {len(cases_df)}")
    print(f"- Mean Retinal Saliency:   {cases_df['retinal_energy_fraction'].mean()*100:.2f}%")
    print(f"- Non-Retinal Leakage:     {avg_leakage*100:.2f}%")
    print(f"- Checkpoint Integrity:    PASSED ({ckpt_sha})")
    print("=" * 70)

if __name__ == "__main__":
    main()
