# End-to-End Inference Pipeline Specification (Phase E015)

## 1. Architectural Pipeline Flow

[Raw Fundus Input (BGR / RGB / PIL / Path)]
                     |
                     v
     [Input Sanitization & Validation]
     (NaN/Inf checks, channel alignment)
                     |
                     v
       [Deterministic IQA Safety Gate]
                     |
     +---------------+---------------+
     |                               |
     v                               v
[IQA FAIL]                      [IQA PASS]
     |                               |
     |                               v
     |              [E001 Baseline Transform (512x512)]
     |                               |
     |                               v
     |              [Frozen E007 Model (EfficientNet-B0)]
     |                               |
     |                               v
     |              [Temperature Calibration (T = 0.7785)]
     |                               |
     |                               v
     |              [Referable Triage Decision (Grade >= 2)]
     |                               |
     |                               v
     |              [Grad-CAM Attribution (features[-1])]
     |                               |
     v                               v
[Recapture / Human Review]  [Clinical Screening Report]

---

## 2. Canonical Schema Specification
Every pipeline execution returns a standardized dictionary with the following schema:

* image_id: String identifier for encounter tracking.
* quality:
  * status: PASS or FAIL.
  * metrics: Deterministic foreground-masked features (fov_coverage, laplacian_variance, edge_density, mean_intensity, dark_fraction, bright_fraction, percentile_spread_90, noise_mad).
  * failed_checks: List of string rejection reasons.
  * message: Patient/operator recapture guidance.
* classification: None if IQA fails; otherwise:
  * predicted_grade: Discrete ICDR severity grade (0 to 4).
  * class_probabilities: Calibrated 5-class probability array.
  * referable: Boolean flag (Grade >= 2).
  * referable_probability: Sum of calibrated probabilities for Grades 2, 3, and 4.
  * confidence: Maximum class probability.
* calibration:
  * temperature: 0.7785.
  * calibrated: Boolean flag.
  * method: Explicit label stating "Frozen development calibration from APTOS validation."
* explanation:
  * gradcam_available: Boolean status flag.
  * target_layer: model.model.features[-1].
  * status: generated, skipped, or unavailable.
  * heatmap: 2D normalized floating-point array [0, 1].
  * overlay: Blended RGB uint8 visualization.
* recommendation:
  * action: Categorical operational triage code.
  * reason: Diagnostic rationale.
* metadata:
  * model_version: E007.
  * model_checkpoint_sha256: SHA-256 fingerprint.
  * timestamp: UTC ISO 8601 string.
  * pipeline_version: Pipeline release tag.
  * scientific_mandate: Regulatory statement.

---

## 3. Triage Escalation Protocol
* IQA_FAIL -> RECAPTURE_OR_HUMAN_REVIEW: Fails Tier 1 or Tier 2 checks. Model forward pass is blocked.
* REFERABLE (Grade >= 2) -> SPECIALIST_REFERRAL: Identifies referable DR based on the project boundary.
* LOW_CONFIDENCE (Confidence < 0.60) -> HUMAN_REVIEW: Flags marginal predictions. Documented engineering placeholder; not clinically validated.
* NON_REFERABLE (Grade < 2) -> ROUTINE_MONITORING: In-bounds mild or normal findings subject to screening workflow.

---

## 4. Failure Mode Handling
* Array Anomalies: NaN, Inf, or invalid channel counts (!= 1, 3, 4) are caught during input validation and route to RECAPTURE_OR_HUMAN_REVIEW.
* Missing Checkpoints: Explicit path and SHA-256 validation prevents execution with missing or altered weights.
* Grad-CAM Degradation: Failures during backward attribution hooks log warnings and set explanation.status = "unavailable" without corrupting the prediction payload.
