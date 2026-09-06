# Experiment E006: Cumulative Ordinal Reformulation

## Executive Status: REJECTED (Validation Only)

### Canonical Performance Metrics (Active Validation Cohort, N=601)
*Evaluated under native training validation pipeline (`create_e006_data_loaders`)*

| Metric | Canonical E006 Result | Project Target / E004 Historical Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.7537** | 0.8531 (E004) | Below Benchmark |
| **Macro F1** | **0.5970** | 0.7581 (E004) | Below Benchmark |
| **Quadratic Weighted Kappa (QWK)** | **0.8914** | 0.8876 (E004) | Parity Achieved |
| **Referable Sensitivity (Grade >= 2)** | **0.8730** | > 0.9000 (Target) / 0.9423 (E004) | **FAILED Safety Target** |
| **Referable Specificity** | **0.9456** | > 0.9000 (Target) / 0.9423 (E004) | Passed |
| **Active Validation Samples** | **601** | 572 (Historical E004) | Verified Clean Cohort |

> **Forensic Audit Disclosure**: A standalone script previously yielded an artificial Macro F1 of 0.2556. Forensic inspection confirmed that result was caused by an input transform mismatch (bypassing native contour masking). Checkpoint metadata confirms the canonical authentic validation metric is Macro F1 = **0.5970**.

### Reason for Rejection
1. **Clinical Sensitivity Failure**: 87.30% referable sensitivity falls below the mandatory >90% clinical screening threshold.
2. **Performance Deficit**: Macro F1 (0.5970) remains below the historical categorical baseline (0.7581).
3. **Preprocessing Brittleness**: The ordered scalar projection exhibits extreme sensitivity to input margin variations.
