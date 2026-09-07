# Data Leakage & Patient Partitioning Risk Assessment
**Phase 2 Audit Finding | Project SIH26038**

### 1. Risk Vector Analysis
Data leakage across partitions can produce over-optimistic performance estimates. Four specific vectors were audited:

Vector 1: Bilateral Patient Leakage (OD in Train, OS in Test) ──> Prevented by Patient ID grouping
Vector 2: Near-Duplicate / Rapid Burst Capture Leakage      ──> Prevented by pHash cluster co-location
Vector 3: External Benchmark Contamination                  ──> Prevented by physical directory quarantine
Vector 4: Preprocessing Parameter Leakage                    ──> Prevented by training-only parameter fitting

---

### 2. Specific Dataset Leakage Findings

#### A. APTOS 2019
* **Finding**: Direct patient identifiers are absent from dataset metadata.
* **Perceptual Hash Audit**: `dHash` and `pHash` analyses identified 18 clusters of near-duplicate images (Hamming distance $\le 3$).
* **Mitigation for Phase 3**: These 18 clusters (encompassing 38 individual images) are assigned explicitly to `data/dataset_audit/duplicate_candidates.csv`. The Phase 3 splitting pipeline must treat each cluster as a single indivisible block, ensuring no pair crosses the Train/Val/Test boundary.

#### B. EyePACS
* **Finding**: 35,126 images share 17,563 distinct patient IDs, with OD and OS images present for each subject.
* **Mitigation for Phase 3**: Stratified Group K-Fold splitting must be enforced on `patient_id`. Image-level random splitting is strictly prohibited.

#### C. Messidor-2
* **Finding**: 1,748 images represent 874 bilateral pairs.
* **Mitigation**: Dataset is quarantined under `data/external/messidor2/` and excluded from model development splits.

---

### 3. Leakage Guardrail Checklist for Phase 3
- [x] Exact bitwise duplicates identified via SHA-256 (0 cross-dataset duplicates found).
- [x] Near-duplicate capture pairs indexed via perceptual hashing for clustered partitioning.
- [x] Group-splitting rule established for multi-image patient records.
- [x] External validation set isolated from the training pipeline.

