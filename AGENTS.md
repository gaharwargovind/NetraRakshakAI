# Autonomous Agent & Engineer Guidelines
**Project:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India

### 1. Fundamental Principles
1. **Never Silently Modify a Clinical Requirement or Protocol**: Any change to clinical criteria, referral logic, or experimental design must be explicitly logged, justified, and approved.
2. **Strict Ground-Truth Honesty**: Never fabricate, estimate, or assume performance metrics. If an evaluation has not been conducted, state: *Not Yet Evaluated*.
3. **Clinical Terminology Guardrails**:
   * Use **AI Prediction** for raw model classification outputs.
   * Use **Screening Referral Recommendation** for automated triage decisions.
   * Reserve **Clinical Diagnosis** exclusively for human ophthalmologists.
4. **Target Criteria vs. Achieved Results**: Sensitivity $\ge 90\%$, Specificity $\ge 85\%$, and $\text{ECE} \le 0.05$ are Target Acceptance Criteria, not verified achievements[cite: 1].

---

### 2. Engineering & Code Standards
* **Python Runtime**: Strictly Python 3.11+.
* **Path Resolution**: Always use `pathlib.Path`. Never use hardcoded strings or machine-specific paths (`/Users/...`, `C:\...`).
* **Configuration-Driven**: All runtime parameters, hyper-parameters, thresholds, and seeds must be specified via `configs/*.yaml`.
* **Type Annotations & Documentation**: All public functions and classes must include Python type hints and standard docstrings.
* **Logging Standard**: Use Python's `logging` module. Never leave raw `print()` statements in production or library code.
* **Implementation Integrity**: Placeholders must explicitly raise `NotImplementedError("Phase X required")` rather than faking execution.

---

### 3. Machine Learning Governance
* **Data Leakage Isolation**:
  * Patient identifiers must never cross partitions. All images for a patient belong to one split.
  * Validation and Test sets must never be augmented.
  * Preprocessing statistics (mean, std) must be derived strictly from the Training partition.
* **Controlled Experiments**: Follow the ladder E001–E010 sequentially. Modify exactly **one** independent variable per experiment.
* **Model Selection Rule**: Model selection, hyperparameter tuning, and threshold selection must be completed on the Validation partition. The Internal Test partition is run once. The External Validation set remains frozen until final evaluation.

---

### 4. Standard Operational Workflow
For every assigned task, the agent must execute:
1. **Inspect**: Review directory state, configuration files, and documentation.
2. **Plan**: Outline architectural boundaries, interfaces, and testing strategies.
3. **Implement**: Write modular, typed, clean, reproducible code.
4. **Test**: Verify syntax, run pytest test suites, and inspect coverage.
5. **Review**: Ensure no clinical boundaries or leakage rules were violated.
6. **Report**: Document files created, files modified, test logs, and next steps.