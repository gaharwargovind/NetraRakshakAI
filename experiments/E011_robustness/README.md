# Experiment E011: Robustness Stress Test (E007 Model)

## Executive Summary
- **Status:** COMPLETED (Non-training diagnostic stress test)
- **Model Checkpoint:** `models/checkpoints/E007_best_model.pt` (Active Champion)
- **Calibration Layer:** $T = 0.7785$ (Frozen post-hoc scaling)
- **Cohort:** $N = 601$ (`data/processed/aptos/validation.csv`)

## Clean Baseline Performance
- **Macro F1:** `0.7123`
- **QWK:** `0.9156`
- **Referable Sensitivity:** `0.9444` (94.44%)
- **Referable Specificity:** `0.9427` (94.27%)

## First Clinical Failure Points (Sensitivity < 90%)
{
  "gaussian_blur": {
    "first_failure_severity": null,
    "note": "Remained >= 90% sensitivity across all tested levels"
  },
  "gaussian_noise": {
    "first_failure_severity": 2,
    "condition": "Mild sensor noise (sigma=15)",
    "sensitivity": 0.5952,
    "macro_f1": 0.2844
  },
  "brightness": {
    "first_failure_severity": null,
    "note": "Remained >= 90% sensitivity across all tested levels"
  },
  "contrast": {
    "first_failure_severity": 1,
    "condition": "Severe contrast loss (-60%)",
    "sensitivity": 0.869,
    "macro_f1": 0.5515
  },
  "jpeg_compression": {
    "first_failure_severity": null,
    "note": "Remained >= 90% sensitivity across all tested levels"
  }
}
