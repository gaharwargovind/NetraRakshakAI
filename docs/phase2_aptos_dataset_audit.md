# Phase 2 - APTOS 2019 Dataset Audit

## Status

Phase: 2 - Dataset Acquisition and Audit
Dataset: APTOS 2019 Blindness Detection
Status: COMPLETE
Date: 2026-09-05

## 1. Acquisition

APTOS training data has been downloaded and stored locally.

Location:

data/raw/aptos/
- train.csv
- train_images/
- aptos_conflicting_duplicate_groups.csv

CSV records: 3662
PNG images: 3662

## 2. CSV and Image Matching

CSV records: 3662
Image files: 3662
Missing images: 0
Extra images: 0

Every training record has a corresponding image.

## 3. Class Distribution

| Diagnosis | Meaning | Count |
|---|---|---:|
| 0 | No DR | 1805 |
| 1 | Mild NPDR | 370 |
| 2 | Moderate NPDR | 999 |
| 3 | Severe NPDR | 193 |
| 4 | Proliferative DR | 295 |
| Total | | 3662 |

The dataset is class-imbalanced.

## 4. Referable DR

For the MVP screening workflow:

diagnosis >= 2

Non-referable: grades 0-1
Referable: grades 2-4

Non-referable records: 2175
Referable records: 1487

This is the project's operational MVP definition and does not resolve the separate DME requirement.

## 5. Image Integrity

Total PNG images: 3662
Valid images: 3662
Corrupt images: 0

All audited images were readable.

## 6. Image Properties

All audited images are RGB PNG files.

The images have heterogeneous dimensions.

Observed width range: 474-4288 pixels
Observed height range: 358-2848 pixels

Therefore preprocessing and standardized resizing are required.

## 7. Exact Duplicate Audit

SHA-256 content audit:

Total image records: 3662
Unique image contents: 3534
Exact duplicate groups: 123
Images involved in duplicate groups: 251
Duplicate occurrences beyond unique contents: 128

## 8. Conflicting Exact Duplicates

Of the 123 exact duplicate groups:

Same-label groups: 93
Conflicting-label groups: 30
Records in conflicting groups: 62

The affected records are stored in:

data/raw/aptos/aptos_conflicting_duplicate_groups.csv

The manifest has been verified:

Conflict groups: 30
Affected records: 62
Unique affected images: 62
Missing affected images: 0

These records are NOT automatically considered mislabeled.

The correct interpretation is:

Byte-identical image content has been assigned different labels in the source metadata. Ground-truth provenance is unresolved.

## 9. Duplicate Handling Policy

Raw images and source labels will not be modified.

Conflicting exact-duplicate groups are quarantined from model development until an explicit handling policy is established.

Consistent exact duplicates may remain in the raw dataset, but identical image content must never cross train, validation, or test boundaries.

All members of an exact duplicate group must belong to the same split.

## 10. Leakage Prevention

The following rules apply:

1. The test set must remain untouched during model development.
2. Exact duplicate groups are indivisible during splitting.
3. Identical image content cannot occur in multiple splits.
4. Conflicting duplicate groups are excluded from development splitting.
5. Random image-level splitting without duplicate grouping is prohibited.

## 11. Dataset Limitations

APTOS is primarily a DR severity classification dataset.

It does not provide sufficient DME-specific labeling for the complete DME referral requirement.

It also does not provide all lesion-level annotations required by the complete SIH specification.

Therefore additional datasets and external validation are required in later phases.

## 12. Dataset Role

APTOS will be used primarily for training and development of the five-class DR severity model.

A separate external dataset will be required for external validation.

The external validation dataset must not be used for tuning or fine-tuning.

## 13. Phase 2 Checklist

[x] APTOS metadata acquired
[x] APTOS images acquired
[x] CSV/image correspondence verified
[x] Image integrity audited
[x] Class distribution measured
[x] Image dimensions audited
[x] Exact duplicates identified
[x] Conflicting duplicate groups identified
[x] Conflict manifest created
[x] Conflict manifest verified
[x] Leakage-prevention policy documented

[ ] Development split created
[ ] External validation dataset acquired

## Conclusion

APTOS acquisition and initial dataset integrity auditing are complete.

The dataset can proceed to Phase 3 provided that duplicate-aware splitting and conflicting-label quarantine are enforced.

PHASE 2 STATUS: COMPLETE
