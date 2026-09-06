# E006 Discrepancy Forensic Audit

## Executive Finding
An apparent performance collapse occurred when evaluating the verified checkpoint `models/checkpoints/E006_best_model.pt` across two distinct execution harnesses:
- **Authentic Training Validation (Epoch 15)**: Macro F1 = **0.5970**, Accuracy = **75.37%**, QWK = **0.8914**, Referable Sensitivity = **87.30%**, Referable Specificity = **94.56%** ($N = 601$).
- **Standalone Re-Evaluation Harness**: Macro F1 = **0.2556**, Accuracy = **41.43%**, QWK = **0.2205**, Referable Sensitivity = **53.97%**, Referable Specificity = **67.05%** ($N = 601$).

Forensic inspection of checkpoint weights, internal metadata, and data loading code demonstrates that:
1. The checkpoint metadata confirms an authentic validation peak of **`best_metric: 0.5970168685238477`** achieved at **`epoch: 15`** on **`expected_active_val_count: 601`**.
2. The standalone evaluation result ($Macro F1 = 0.2556$) is **INVALID and SUPERSEDED**. The standalone harness applied PIL-based circular border cropping and direct squashing, bypassing the exact OpenCV contour masking and letterbox aspect-ratio padding implemented in `src/data/loaders.py` (`create_e006_data_loaders`).
3. Unlike categorical classification with Softmax (which is scale-invariant to logit shifts), E006's `OrderedThresholdHead` projects all 1,280 feature channels into a single scalar $f(x)$ evaluated against fixed thresholds:
   $$\theta_1 = 0.0014, \quad \theta_2 = 1.1242, \quad \theta_3 = 2.3925, \quad \theta_4 = 3.5982$$
   Differences in black-border ratios shifted scalar activations systematically past cutoff boundaries, pushing 41 healthy retinas (Grade 0) into Grade 4 and collapsing standalone performance.

## Path Tracing & Component Comparison

| Component | Authentic Validation Pipeline (`train.py` / `loaders.py`) | Standalone Script (`evaluate_e006_current_validation.py`) | Audit Status |
| :--- | :--- | :--- | :--- |
| **Model Weights** | In-memory / checkpoint state at Epoch 15 | Loaded from `models/checkpoints/E006_best_model.pt` | **IDENTICAL** |
| **Backbone Architecture** | `DROrdinalNet` (EfficientNet-B0 + `OrderedThresholdHead`) | `DROrdinalNet` (EfficientNet-B0 + `OrderedThresholdHead`) | **IDENTICAL** |
| **Ordinal Decoding Rule** | Monotonic cumulative count ($\tau = 0.5$) | Monotonic cumulative count ($\tau = 0.5$) | **IDENTICAL** |
| **Input Transform Pipeline** | `create_e006_data_loaders` (OpenCV contour crop + letterbox padding) | Standalone PIL crop + raw resize | **MISMATCH (Root Cause)** |
| **Cohort Manifest** | `data/processed/aptos/validation.csv` ($N = 601$) | `data/processed/aptos/validation.csv` ($N = 601$) | **IDENTICAL** |
| **Ground Truth Labels** | `diagnosis` integers 0..4 | `diagnosis` integers 0..4 | **IDENTICAL** |
| **Metric Formulation** | Macro F1, Accuracy, QWK, Referable Sens/Spec | Macro F1, Accuracy, QWK, Referable Sens/Spec | **IDENTICAL** |

## Scientific Status & Final Verdict
- **Canonical E006 Metrics**: Macro F1 = **0.5970**, Accuracy = **75.37%**, QWK = **0.8914**, Referable Sensitivity = **87.30%**, Referable Specificity = **94.56%**.
- **Final Verdict**: **REJECTED**.
  - Referable sensitivity (87.30%) fails the clinical safety threshold (>90%).
  - Macro F1 (0.5970) remains significantly below the retained historical E004 categorical benchmark (0.7581).
  - The architecture demonstrates extreme sensitivity to minor preprocessing boundary variances.
