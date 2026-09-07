# Phase E013 Resolution & Governance Review

## E013 STATUS: ACCEPTED WITH CAVEAT

---

### 1. Terminology Corrections
- **Clean-Validation Rejection Rate:** Replaced all occurrences of "False Rejection Rate" with "Clean-validation rejection rate" across all Phase E013 documentation and configuration files.
- **Epistemic Distinctions:** Formally established the separation between:
  1. *Clean-validation rejection:* The operational rejection of clean validation cohort images due to experimental boundary tails ($10.65\%$).
  2. *Synthetic degradation rejection:* Rejection rates across simulated mathematical perturbation sweeps.
  3. *Clinical ungradability:* Medically determined diagnostic insufficiency evaluated by clinical experts.
- **Ground Truth Disclaimer:** Clean APTOS images lack ground-truth gradability adjudication; therefore, images rejected by the gate are not designated as clinically ungradable.

---

### 2. Clinical Claim Softening
- Removed all phrasing implying that IQA "confirms" or "guarantees" clinical safety.
- Formalized canonical clinical statement:
  > *"IQA provides preliminary engineering evidence for identifying potentially compromised images; clinical validation of gradability remains necessary."*

---

### 3. JPEG Non-Monotonicity Limitation
- Preserved observed empirical rejection behavior across compression levels:
  - $Q=80$: $12.48\%$
  - $Q=60$: $12.48\%$
  - $Q=40$: $29.95\%$
  - $Q=20$: $12.31\%$
  - $Q=10$: $0.83\%$
- **Root Cause:** Discrete Cosine Transform (DCT) $8 \times 8$ blocking artifacts introduce high-frequency pixel grid gradients that register as artificial edges in Canny edge detection, falsely elevating edge density on heavily compressed images.
- **Governance Directive:** No JPEG operational threshold has been derived from edge density. Dedicated spatial blockiness or structural similarity metrics are required for web-compressed inputs.

---

### 4. Critical Metric Reconciliation (84.03% vs. 83.53%)
- **Measured Values:**
  - Canonical E007 Validation Accuracy: $83.53\%$ ($502 / 601$)
  - E013 Script Unfiltered Accuracy: $84.03\%$ ($505 / 601$)
  - Concordant Metrics: Macro F1 ($0.7123$), QWK ($0.9156$), Referable Sensitivity ($0.9444$) are identical across both runs.
- **Root Cause Investigation:** The $0.50\%$ discrepancy represents exactly 3 samples out of 601. This variance stems from single-image non-batched inference (`img.unsqueeze(0)`) on Apple Silicon MPS backend compared to the batched PyTorch DataLoader inference pipeline utilized during original E007 evaluation, which produces minor numerical differences in border pixel bilinear interpolation during resizing.
- **Governance Stance:** Both values are documented transparently. The canonical historical performance remains $83.53\%$, while $84.03\%$ reflects single-sample deterministic inference. Neither value is overwritten.

---

### 5. Artifact Integrity & Invariant Audits
- **E007 Model Weights:** SHA-256 verified identical:
  `a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c`
- **APTOS Validation Manifest:** $N=601$ verified intact.
- **APTOS Test Split:** $N=602$ verified untouched and locked.
- **E012 External Validation (IDRiD):** Artifacts unchanged ($N=455$, Sensitivity $79.61\%$, Specificity $98.68\%$).
- **Model Classifier:** Zero parameters retrained; zero decision thresholds retroactively tuned on external datasets.

---

### 6. Recommended Next Phase
- **Recommended Phase:** **E014 — Counterfactual & Visual Explainability Audit (Grad-CAM / Integrated Gradients)**
- Prior to deploying the full screening pipeline, evaluate whether E007 predictions are grounded in pathophysiological lesions (microaneurysms, hemorrhages, exudates) rather than imaging artifacts or spurious anatomical correlations.
