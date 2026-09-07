import os, sys, json, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT_DIR = Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.inference.predictor import ScreeningPredictor, EXPECTED_E007_SHA
from src.inference.report_generator import compile_screening_report

def main():
    print("=" * 70)
    print("       EXPERIMENT E015: END-TO-END INFERENCE BENCHMARK")
    print("=" * 70)

    val_csv = ROOT_DIR / "data/processed/aptos/validation.csv"
    val_df = pd.read_csv(val_csv)
    aptos_img_dir = ROOT_DIR / "data/raw/aptos/train_images"

    sample_ids = val_df["id_code"].head(10).tolist()
    predictor = ScreeningPredictor()
    print(f"[*] Loaded ScreeningPredictor on device: {predictor.device}")

    latencies = {
        "input_sanitization": [],
        "iqa_gate": [],
        "preprocessing": [],
        "e007_inference": [],
        "gradcam": [],
        "total_pipeline": []
    }

    records = []
    out_dir = ROOT_DIR / "experiments/E015_inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warmup pass
    warmup_path = aptos_img_dir / f"{sample_ids[0]}.png"
    _ = predictor.predict(warmup_path, generate_saliency=True)

    for i, id_c in enumerate(sample_ids):
        img_path = aptos_img_dir / f"{id_c}.png"

        t0 = time.perf_counter()
        bgr_np, pil_img = predictor._sanitize_input(img_path)
        t1 = time.perf_counter()

        gate_res = predictor.quality_gate.evaluate(bgr_np)
        t2 = time.perf_counter()

        input_tensor = predictor.transform(pil_img).unsqueeze(0).to(predictor.device)
        t3 = time.perf_counter()

        with torch.no_grad():
            logits = predictor.model(input_tensor)
            probs = torch.softmax(logits / predictor.temperature, dim=-1)
        t4 = time.perf_counter()

        input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
        _, _, _ = predictor.gradcam.generate(input_tensor_grad, target_class=int(torch.argmax(probs).item()))
        t5 = time.perf_counter()

        res = predictor.predict(img_path, image_id=id_c, generate_saliency=True)
        t6 = time.perf_counter()

        latencies["input_sanitization"].append((t1 - t0) * 1000)
        latencies["iqa_gate"].append((t2 - t1) * 1000)
        latencies["preprocessing"].append((t3 - t2) * 1000)
        latencies["e007_inference"].append((t4 - t3) * 1000)
        latencies["gradcam"].append((t5 - t4) * 1000)
        latencies["total_pipeline"].append((t6 - t0) * 1000)

        pred_grade = res["classification"]["predicted_grade"] if res["classification"] else None
        ref_status = res["classification"]["referable"] if res["classification"] else None
        conf = res["classification"]["confidence"] if res["classification"] else None

        records.append({
            "image_id": id_c,
            "quality_status": res["quality"]["status"],
            "predicted_grade": pred_grade,
            "referable": ref_status,
            "confidence": conf,
            "action": res["recommendation"]["action"],
            "total_latency_ms": round((t6 - t0) * 1000, 2)
        })
        print(f"  [{i+1:2d}/10] Image {id_c} -> Quality: {res['quality']['status']} | Grade: {pred_grade} | Total: {records[-1]['total_latency_ms']:.1f} ms")

    bench_summary = {k: {"mean_ms": round(float(np.mean(v)), 2), "std_ms": round(float(np.std(v)), 2)} for k, v in latencies.items()}

    # Sample reports
    sample_res = predictor.predict(aptos_img_dir / f"{sample_ids[0]}.png", image_id="sample_encounter_01")
    compile_screening_report(sample_res, out_dir / "sample_report.pdf")
    compile_screening_report(sample_res, out_dir / "sample_report.txt")

    metrics_payload = {
        "experiment_id": "E015",
        "name": "End-to-End Inference Pipeline Integration",
        "benchmark_n": len(sample_ids),
        "latency_breakdown": bench_summary,
        "frozen_invariants": {
            "e007_checkpoint_sha256": predictor.checkpoint_sha,
            "temperature": predictor.temperature,
            "iqa_threshold_policy": "Tier 2 Experimental (APTOS validation calibrated)",
            "referable_decision_boundary": "predicted_grade >= 2"
        },
        "governance_mandate": "AI-assisted screening tool; benchmark latencies reflect single-sample local execution and do not guarantee clinical deployment throughput."
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    config_yaml = f"""experiment_id: E015_inference
phase: PHASE_E015_END_TO_END_INFERENCE
base_model: E007
checkpoint:
  path: models/checkpoints/E007_best_model.pt
  sha256: {EXPECTED_E007_SHA}
calibration:
  temperature: 0.7785
  status: FROZEN_DEVELOPMENT
decision_rule:
  referable: "predicted_grade >= 2"
quality_gate:
  status: EXPERIMENTAL_ENGINEERING
triage_actions:
  - RECAPTURE_OR_HUMAN_REVIEW
  - HUMAN_REVIEW
  - SPECIALIST_REFERRAL
  - ROUTINE_MONITORING
"""
    (out_dir / "config.yaml").write_text(config_yaml)

    readme_content = f"""# Experiment E015: End-to-End Inference Pipeline

## Executive Summary
- **Status:** COMPLETE
- **Pipeline Architecture:** `Input Sanitization -> IQA Quality Gate -> Frozen E007 Inference -> Temperature Calibration -> Grad-CAM Saliency -> Triage Escalation`
- **Mean Pipeline Latency:** `{bench_summary['total_pipeline']['mean_ms']:.1f} ± {bench_summary['total_pipeline']['std_ms']:.1f} ms` (Device: `{predictor.device}`)
- **IQA Intercept Latency:** `{bench_summary['iqa_gate']['mean_ms']:.1f} ms`
- **Model Inference Latency:** `{bench_summary['e007_inference']['mean_ms']:.1f} ms`

## Governance & Safety Mandate
"AI-assisted diabetic retinopathy screening; clinical validation of gradability and diagnostic adjudication remain necessary."
"""
    (out_dir / "README.md").write_text(readme_content)

    print("\n" + "=" * 70)
    print("       E015 BENCHMARK COMPLETE")
    print("=" * 70)
    for k, v in bench_summary.items():
        print(f"  - {k:<22}: {v['mean_ms']:6.2f} ms ± {v['std_ms']:5.2f} ms")
    print(f"[*] Verified Checkpoint SHA: {predictor.checkpoint_sha}")
    print("=" * 70)

if __name__ == "__main__":
    main()
