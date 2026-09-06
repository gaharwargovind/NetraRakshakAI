# Validation Protocol & Clinical Metrics Specification
**Document Version:** 3.0 (Updated Phase 6 — Evaluation Framework)  
**Project:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India  

---

### 1. Data Partitioning Governance & Split Roles

| Partition | Allocation | Role & Governance Mandate |
| :--- | :--- | :--- |
| **Train Split** | 70% (2,563 images) | Model parameter gradient optimization and data augmentation exploration. Preprocessing statistics ($\mu, \sigma$) must be derived exclusively from this split. |
| **Validation Split** | 15% (549 images) | Hyperparameter tuning, model architecture selection, early stopping, decision threshold tuning ($\tau_{\text{ref}}$), and temperature scaling calibration ($T$). |
| **Internal Test Split** | 15% (550 images) | **LOCKED BENCHMARK**. Evaluated strictly once per completed experiment. Never used for model selection, parameter tuning, or early stopping. |
| **External Validation** | Messidor-2 (1,748 images) | **QUARANTINED GENERALIZATION BENCHMARK**. Evaluated frozen in Phase 12. No tuning, threshold adjustment, or retraining is permitted. |

---

### 2. Clinical Classification Standards

#### Five-Class ICDR Severity Scale
* **Grade 0 (No DR)**: Absence of microaneurysms, hemorrhages, or exudates.
* **Grade 1 (Mild NPDR)**: Microaneurysms only.
* **Grade 2 (Moderate NPDR)**: More than microaneurysms, but less than severe NPDR.
* **Grade 3 (Severe NPDR)**: Meets $\ge 1$ criterion of the 4-2-1 rule, no proliferative features.
* **Grade 4 (Proliferative DR - PDR)**: Neovascularization and/or vitreous/preretinal hemorrhage.

#### Operational Referable DR Triage
* **Referable DR**: Defined as **ICDR Grade $\ge 2$** (Moderate NPDR, Severe NPDR, or PDR).
* **Non-Referable Cohort**: ICDR Grade 0 and Grade 1 (scheduled for routine PHC follow-up in 12 months).

---

### 3. Evaluation Metrics Hierarchy

#### Primary Clinical Screening Metrics
* **Referable Sensitivity**:
  $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Target Acceptance Criterion*: $\ge 90.0\%$[cite: 1].
* **Referable Specificity**:
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
  *Target Acceptance Criterion*: $\ge 85.0\%$[cite: 1].

#### Secondary Multi-Class Metrics
* **Quadratic Weighted Kappa (QWK)**: Measures ordinal agreement while penalizing distant misclassifications:
  $$\kappa = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(N - 1)^2}$$
* **Macro-Averaged F1-Score**: Equal-weighted mean of F1 across all five classes, preventing severe classes from being masked by class imbalance.
* **Per-Class Precision, Recall, and F1**: Evaluated across each individual grade (0 to 4).
* **Confusion Matrices**: Both absolute counts and row-normalized (true class recall) formats.

#### Probabilistic Metrics
* **One-vs-Rest ROC-AUC**: Macro and weighted averages across the 5 classes.
* **Referable ROC-AUC**: Evaluates discrimination using $P(\text{Referable}) = \sum_{k=2}^4 P(\text{Grade}=k)$.

---

### 4. Target Acceptance Criteria vs. Observed Results

Target metrics are clinical design benchmarks, not guaranteed algorithmic results[cite: 1]:

| Metric | Target Acceptance Criterion[cite: 1] | E001 Baseline Test Result | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Referable Sensitivity** | $\ge 90.0\%$[cite: 1] | **86.88%** | Target Not Achieved |
| **Referable Specificity** | $\ge 85.0\%$[cite: 1] | **92.81%** | Target Achieved |
| **Expected Calibration Error (ECE)** | $\le 0.05$[cite: 1] | Not Yet Evaluated (Phase 10) | Pending Phase 10 |

*Clinical Caveat*: The E001 baseline is an initial reference experiment and is **not** a clinically validated diagnostic device[cite: 1].