# Experiment E010: E007 Post-Hoc Confidence Calibration

## Objective & Governance
- **Status:** COMPLETED & VALIDATED
- **Type:** Post-hoc calibration layer (No retraining; no weights altered).
- **Target Model:** `models/checkpoints/E007_best_model.pt` (Active Champion).
- **Validation Split:** `data/processed/aptos/validation.csv` ($N = 601$).
- **Held-Out Test Set:** Completely unaccessed.

## Key Findings
- **Fitted Temperature:** $T = 0.7785$
- **Multiclass ECE:** Reduced from **7.97%** to **3.77%** ($\Delta = -0.0420$)
- **Multiclass NLL:** Reduced from **0.4736** to **0.4533** ($\Delta = -0.0203$)
- **Referable Risk ECE:** Reduced from **4.38%** to **3.59%**
- **Invariance Guarantee:** All 601 argmax predictions, accuracy (0.8403), sensitivity (0.9444), and confusion matrix entries are byte-for-byte identical.
