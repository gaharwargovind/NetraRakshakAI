# E006 Standalone Validation Artifacts

## Protocol Status: SUPERSEDED / INVALID (Preserved for Forensic Audit Only)

### Critical Disclosure
The metrics contained in `metrics.json` within this directory (Accuracy: 0.4143, Macro F1: 0.2556, QWK: 0.2205) are **INVALID** and **SUPERSEDED**.

### Root Cause of Artifact Invalidation
- The standalone harness `scripts/evaluate_e006_current_validation.py` applied a standard PIL crop and squashing resize, bypassing the OpenCV contour isolation and letterbox padding implemented in `src/data/loaders.py` (`create_e006_data_loaders`).
- Because E006 uses an `OrderedThresholdHead` that maps features to a single scalar against rigid cutoffs, this preprocessing difference caused severe out-of-distribution domain shift.
- The authentic, canonical E006 validation performance is documented in `experiments/E006_ordinal/README.md` and verified in checkpoint metadata as:
  - **Macro F1**: **0.5970**
  - **Accuracy**: **0.7537**
  - **QWK**: **0.8914**
  - **Referable Sensitivity**: **0.8730**
  - **Referable Specificity**: **0.9456**
