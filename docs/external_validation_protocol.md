# External Validation Protocol (Phase E012)

## 1. Frozen Evaluation Invariants
- **Model Checkpoint:** `models/checkpoints/E007_best_model.pt` (Frozen, SHA-256 verified)
- **Architecture:** `DREfficientNet` (EfficientNet-B0 with 5-class categorical Softmax)
- **Calibration:** Frozen Temperature Scaling ($T = 0.7785$ derived from E010)
- **Preprocessing:** Byte-for-byte identical `E001BaselineTransform` ($512 	imes 512$, deterministic circular crop, ImageNet normalization)
- **Tuning Prohibitions:** No fine-tuning, no threshold adjustments, and zero access to APTOS `test.csv`.

## 2. Cohort Boundary Disambiguation
- **Internal Validation Cohort:** APTOS 2019 validation split ($N=601$, used for model selection and calibration fitting).
- **Held-Out Test Cohort:** APTOS 2019 test split ($N=602$, locked benchmark, strictly isolated).
- **External Validation Cohort:** IDRiD Disease Grading cohort ($N=516$, independent hospital/camera cohort).

## 3. Target Metric Hierarchy
- **Primary Clinical Safety Gate:** Referable Sensitivity (Grade $\\ge 2$) $\\ge 90.00\\%$
- **Primary Discrimination Metric:** Macro F1 Score
- **Secondary Clinical Agreement:** Quadratic Weighted Kappa (QWK)
- **Secondary Metrics:**
  - Referable Specificity
  - Overall Accuracy
  - Mean Absolute Error (MAE)
  - Adjacent-Grade Error ($|\\hat{y} - y| = 1$)
  - Severe-Grade Error ($|\\hat{y} - y| \\ge 2$)
  - Multiclass Expected Calibration Error (ECE) with frozen $T=0.7785$
  - Referable Risk ECE with frozen $T=0.7785$
