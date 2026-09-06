# Validation Partition Integrity & Exclusion Audit Report
**Project:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Phase:** Phase 7 Diagnostic Audit  
**Status:** NON-DESTRUCTIVE AUDIT COMPLETE — ROOT CAUSE IDENTIFIED  

---

### 1. Problem Statement
The experiment logs and confusion matrices for **E005** (`experiments/E005_architecture/metrics.json`) reported an evaluated sample count summing to **572** validation images for all three candidate backbones:
* EfficientNet-B0: $284 + 14 + 10 + 0 + 0 + 8 + 37 + 11 + \dots + 23 = \mathbf{572}$
* ResNet-50: $280 + 16 + 12 + \dots + 22 = \mathbf{572}$
* ConvNeXt-Tiny: $282 + 15 + 11 + \dots + 23 = \mathbf{572}$

However, the file `data/processed/aptos/validation.csv` contains **601** row records. A discrepancy of **29** records was identified between the split manifest line count and the evaluated validation sample count.

---

### 2. Quantitative Summary
* **Expected Total Validation Records (`validation.csv`)**: **601**
* **Observed Evaluation Count in E005**: **572**
* **Difference (Excluded Records)**: **29**
* **Physical Missing Images on Disk**: **0** (All 601 image files exist under `data/raw/aptos/train_images/`)
* **Physical Image Decode/Corruption Errors**: **0** (All 601 image files are verified valid RGB PNG bitstreams)

---

### 3. Root Cause Analysis

The exclusion occurs inside the dataset constructor `APTOSDataset.__init__` in `src/data/loaders.py`:

```python
if "is_conflicting_duplicate" in raw_df.columns:
    filtered_df = raw_df[raw_df["is_conflicting_duplicate"] != True].copy()
    quarantined_count = len(raw_df) - len(filtered_df)
    if quarantined_count > 0:
        logger.info(
            "Quarantined %d conflicting duplicates from %s",
            quarantined_count,
            self.csv_path.name,
        )
    self.df = filtered_df.reset_index(drop=True)