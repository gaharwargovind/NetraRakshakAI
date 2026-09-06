# E004 vs E006 Definitive Resolution Protocol

**Target:** `docs/e004_e006_resolution.md`  
**Status:** Formally Ratified  
**Governance Invariant:** Read-Only Audit (No Retraining, Test Set Locked)  

---

### 1. E004 Checkpoint Status & Benchmark Authority
* An exhaustive filesystem search confirmed that no binary checkpoint file (`.pt`, `.pth`, `.ckpt`) exists for Experiment E004.
* Historical E004 metrics recorded in `experiments/E004_augmentation/metrics.json` (Accuracy: 0.8531, Macro F1: 0.7581, QWK: 0.8876, Sensitivity: 0.9423, Specificity: 0.9423) were evaluated against an unrecoverable 572-image validation cohort.
* The current active validation manifest contains 601 records. At least 15 images from the historical 572 cohort are not in the current validation split.
* **Retraining Policy**: Retraining E004 to reproduce historical numbers is prohibited. Doing so would instantiate a new model subject to non-deterministic optimization stochasticity. E004 metrics remain the project's historical categorical champion benchmark.

### 2. E006 Checkpoint Status & Metric Authentication
* Checkpoint `models/checkpoints/E006_best_model.pt` is intact on disk.
* Checkpoint metadata confirms an authentic validation Macro F1 of **0.597017** at **Epoch 15** across the 601-record cohort.
* Under its native validation transform (`create_e006_data_loaders`), E006 achieved:
  - **Accuracy**: 0.7537 (75.37%)
  - **Macro F1**: 0.5970
  - **QWK**: 0.8914
  - **Referable Sensitivity**: 0.8730 (87.30%)
  - **Referable Specificity**: 0.9456 (94.56%)

### 3. Invalidation of Standalone 0.2556 Evaluation
* The standalone evaluation script `scripts/evaluate_e006_current_validation.py` generated artificial metrics (Macro F1 = 0.2556, Accuracy = 41.43%).
* This drop was caused by input transform domain shift (bypassing native contour cropping and letterbox padding).
* The 0.2556 result is formally marked **INVALID and SUPERSEDED**. It is preserved solely for forensic audit history and must not be cited as E006 model performance.

### 4. Final Scientific Decision
* **Experiment E004**: Retained as historical champion benchmark (Macro F1: 0.7581, Sensitivity: 94.23%).
* **Experiment E006**: **REJECTED**.
  - Fails clinical referable sensitivity gate ($87.30\% < 90.00\%$).
  - Macro F1 ($0.5970$) falls short of the historical E004 benchmark ($0.7581$).
  - Extreme fragility to preprocessing geometry renders cumulative ordinal thresholding unsuitable for deployment.
