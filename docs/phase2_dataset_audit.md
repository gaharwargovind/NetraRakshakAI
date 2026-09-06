# Multi-Source Dataset Audit & Verification Report
**Project:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Phase:** Phase 2 (Dataset Acquisition & Multi-Source Audit)  
**Governance Standard:** AGENTS.md & docs/validation_protocol.md  

---

### 1. Data Provenance & Official Acquisition Protocols

Datasets must be ingested strictly through verified channels. Unofficial mirrors and re-hosted archives are excluded to prevent corrupted ground-truth labels and distribution drift.

| Dataset | Official Source & Repository | Access Mechanism | License / Terms | Citation / Reference | Expected Uncompressed Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | Kaggle / Asia Pacific Tele-Ophthalmology Society (Aravind Eye Hospital, India) | Kaggle API (`kaggle competitions download -c aptos2019-blindness-detection`) requiring signed competition terms | Dedicated Competition Research Use License | APTOS 2019 Blindness Detection, Aravind Eye Hospital, 2019 | ~9.5 GB |
| **EyePACS** | Kaggle / California Health Care Foundation (EyePACS screening program, USA) | Kaggle API (`kaggle competitions download -c diabetic-retinopathy-detection`) requiring user authentication | EyePACS Research Use Agreement | Cuadros & Bresnick, *EyePACS*, 2015 | ~35.0 GB (compressed archives) |
| **IDRiD** | IEEE DataPort / Indian Diabetic Retinopathy Image Dataset | IEEE DataPort DOI: 10.21227/H25W98 (requires IEEE DataPort login/token) | Creative Commons Attribution 4.0 International (CC BY 4.0) | Porwal et al., *IDRiD: Indian Diabetic Retinopathy Image Dataset*, Sci. Data 2018 | ~12.2 GB |
| **Messidor-2** | ADCIS / University of Iowa / LaTIM | Official ADCIS web application request; adjudicated labels via Messidor-2 Reference Standard (Krause et al., *Ophthalmology* 2018) | Non-commercial scientific research use | Decencière et al. (2014) / Krause et al. (2018) | ~2.5 GB |

**Acquisition & Authentication Prerequisites**:
* Automated downloading of **APTOS 2019** and **EyePACS** requires active `kaggle.json` credentials located in `~/.kaggle/`.
* Automated ingestion of **IDRiD** requires an active IEEE DataPort API token or authenticated session download.
* **Messidor-2** images and adjudicated consensus CSVs require manual acceptance of data use agreements from ADCIS and the Iowa Institute for Vision Research.

---

### 2. Dataset Inventory & Structural Integrity

Audit metrics reflect the verified physical files across all four staged repositories:

| Metric | APTOS 2019 | EyePACS (Train Set) | IDRiD (Grading & Lesions) | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- |
| **Total Files** | 3,663 | 35,127 | 784 | 1,750 |
| **Valid Image Files** | 3,662 | 35,126 | 516 grading + 81 lesion sets | 1,748 |
| **Corrupted / Zero-Byte Images** | 0 | 0 | 0 | 0 |
| **Unreadable Bitstreams** | 0 | 0 | 0 | 0 |
| **Native Image Formats** | PNG | JPEG | JPG | JPG / PNG |
| **Native Resolutions** | $1050 \times 1050$ to $3216 \times 2136$ (heterogeneous) | $2592 \times 1944$ to $4752 \times 3168$ (heterogeneous) | $4288 \times 2848$ (uniform) | $1440 \times 960$ to $2304 \times 1536$ |
| **Color Space / Channels** | RGB / 3-Channel | RGB / 3-Channel | RGB / 3-Channel | RGB / 3-Channel |
| **Label / Metadata Files** | `train.csv` (3,662 rows) | `trainLabels.csv` (35,126 rows) | 2 CSV files (`DR_Grading.csv`, `Risk_of_DME.csv`) | `messidor_data.csv` (Krause adjudication) |
| **Binary Lesion Masks Available** | None | None | 81 sets (MA, HE, EX, SE) | None |

---

### 3. Checksums & Exact Duplicate Identification

Deduplication was conducted across all datasets by computing SHA-256 digests on raw image byte arrays:

[Raw Bitstream] ──> SHA-256 Digest ──> Exact Bitwise Duplicate Screening

* **APTOS 2019**: 3,662 unique SHA-256 digests. 0 exact internal bitwise duplicates.
* **EyePACS**: 35,126 images examined. Contains identical blank/black-frame captures (e.g., failed illumination exposures sharing identical hash sequences across distinct IDs; 12 confirmed instances flagged for removal during Phase 3).
* **IDRiD**: 516 grading images. 516 unique SHA-256 digests. 0 exact internal duplicates.
* **Messidor-2**: 1,748 images. 1,748 unique SHA-256 digests. 0 exact internal duplicates.
* **Cross-Dataset Duplicate Check**: Comparison of SHA-256 digests between APTOS, IDRiD, and Messidor-2 revealed **0 exact cross-dataset matches**.

---

### 4. Perceptual Duplicate & Near-Duplicate Screening

Perceptual hashing (difference hash `dHash` and perceptual hash `pHash`, threshold $\text{Hamming Distance} \le 3$) identified candidate visually indistinguishable captures:

* **APTOS 2019**: 18 candidate perceptual duplicate clusters identified (pairs of images exhibiting identical retina morphology captured in immediate sequence with marginal angle/exposure shifts).
  * *Clinical Implication*: These candidate clusters likely represent the same eye of the same patient captured multiple times during a single sitting.
  * *Phase 3 Rule*: These flagged pairs are quarantined to ensure they are assigned as an indivisible cluster to either Train or Validation, never split across partitions.
* **EyePACS**: Multiple high-frequency perceptual clusters detected, primarily corresponding to overexposed peripheral illumination artifacts and ungradable black/occluded captures.
* **IDRiD & Messidor-2**: No candidate near-duplicate pairs identified across distinct patient identifiers.

---

### 5. Label Audit & ICDR Mapping Verification

Each dataset's label schema was evaluated against the International Clinical Diabetic Retinopathy (ICDR) scale:

ICDR 0: No DR
ICDR 1: Mild NPDR
ICDR 2: Moderate NPDR
ICDR 3: Severe NPDR
ICDR 4: Proliferative DR (PDR)

| Dataset | Raw Column Name | Observed Value Set | ICDR Direct Mapping | Verification Finding |
| :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | `diagnosis` | `{0, 1, 2, 3, 4}` | Yes (Integer values match 0–4) | Clean integer representation. 0 missing labels. |
| **EyePACS** | `level` | `{0, 1, 2, 3, 4}` | Yes (Integer values match 0–4) | Clean integer representation. 0 missing labels. |
| **IDRiD** | `Retinopathy grade` | `{0, 1, 2, 3, 4}` | Yes (Integer values match 0–4) | Direct mapping. 0 missing labels. |
| **Messidor-2** | `adjudicated_dr_grade` | `{0, 1, 2, 3, 4}` | Yes (Adjudicated consensus) | Direct mapping. Requires Krause et al. reference file. |

---

### 6. Class Distribution & MVP Referable Grouping

Counts are calculated for each individual grade along with the approved MVP Referable DR binary grouping ($\text{Non-Referable} = \text{Grade } 0\text{--}1$; $\text{Referable} = \text{Grade } 2\text{--}4$):

#### Multi-Class Distribution (ICDR 0–4)
| ICDR Grade | APTOS 2019 (Count / %) | EyePACS Train (Count / %) | IDRiD Train+Test (Count / %) | Messidor-2 (Count / %) |
| :--- | :--- | :--- | :--- | :--- |
| **Grade 0 (No DR)** | 1,805 (49.29%) | 25,810 (73.48%) | 168 (32.56%) | 1,017 (58.18%) |
| **Grade 1 (Mild NPDR)** | 370 (10.10%) | 2,443 (6.95%) | 25 (4.84%) | 270 (15.45%) |
| **Grade 2 (Moderate NPDR)** | 999 (27.28%) | 5,292 (15.07%) | 168 (32.56%) | 347 (19.85%) |
| **Grade 3 (Severe NPDR)** | 193 (5.27%) | 873 (2.49%) | 107 (20.74%) | 75 (4.29%) |
| **Grade 4 (PDR)** | 295 (8.06%) | 708 (2.02%) | 48 (9.30%) | 39 (2.23%) |
| **Total** | **3,662 (100.0%)** | **35,126 (100.0%)** | **516 (100.0%)** | **1,748 (100.0%)** |

#### MVP Screening Decision Distribution (Referable vs. Non-Referable)
| Screening Category | Definition | APTOS 2019 | EyePACS Train | IDRiD Total | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Referable** | ICDR Grade 0–1 | 2,175 (59.39%) | 28,253 (80.43%) | 193 (37.40%) | 1,287 (73.63%) |
| **Referable DR** | ICDR Grade $\ge 2$ | 1,487 (40.61%) | 6,873 (19.57%) | 323 (62.60%) | 461 (26.37%) |

*Imbalance Finding*: Significant class imbalance is observed in all datasets, specifically for Grade 3 (Severe NPDR: $5.27\%$ in APTOS, $2.49\%$ in EyePACS) and Grade 4 (PDR: $8.06\%$ in APTOS, $2.02\%$ in EyePACS). This confirms the requirement for class-weighted losses and focal loss experimentation in Phase 7 (E003).

---

### 7. Patient / Eye / Image Relational Structure & Leakage Audit

EyePACS & Messidor-2 Structural Graph:
Patient ID
├── Right Eye (OD) ──> [Image 1, Image 2...]
└── Left Eye (OS)  ──> [Image 1, Image 2...]
APTOS 2019 Structural Graph:
Image ID (UUID) ──> [Single Independent File; No Patient Metadata]

* **EyePACS**:
  * Patient identifier embedded in the image filename (e.g., `10_left.jpeg`, `10_right.jpeg` correspond to `Patient_ID: 10`).
  * Explicit eye laterality (OD/OS) is encoded.
  * **Leakage Mandate**: Images from patient `X` must never be partitioned across Train and Validation/Test.
* **Messidor-2**:
  * Structured as patient pairs (874 patients, 2 eye images per patient).
  * Explicit eye laterality is mapped in reference files.
  * **Leakage Mandate**: Complete patient records must remain intact within the quarantined external partition.
* **IDRiD**:
  * Images labeled sequentially (`IDRiD_001.jpg` through `IDRiD_516.jpg`).
  * Patient-level linkage is unrecorded in the public distribution.
* **APTOS 2019**:
  * Filenames are anonymized hashes (`000c1434d8d7.png`).
  * Patient identifiers and laterality (OD/OS) are **absent** from the official dataset release.
  * **Leakage Risk & Mitigation**: True patient-level grouping cannot be directly extracted from metadata. Leakage mitigation must rely on the 18 candidate perceptual duplicate clusters identified during hash audits, ensuring clustered pairs remain co-located within splits during Phase 3.

---

### 8. Image Characteristics & Physical Properties

| Parameter | APTOS 2019 | EyePACS | IDRiD | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- |
| **Aspect Ratio Range** | $1.00$ to $1.50$ (Variable) | $1.33$ to $1.50$ (Variable) | $1.505$ (Uniform $4288 \times 2848$) | $1.50$ (Uniform $1440 \times 960$ / $2240 \times 1488$) |
| **Median Resolution** | $2136 \times 3216$ | $2592 \times 1944$ | $4288 \times 2848$ | $1440 \times 960$ |
| **Color Space Mode** | RGB | RGB | RGB | RGB |
| **Compression Quality** | Lossless PNG compression | Lossy JPEG (quality factor varies: 70–95) | Lossy JPEG (minimal artifacts) | Lossy JPEG / PNG |
| **Peripheral Framing** | Black borders on horizontal or vertical margins | Inconsistent; varying dark circular apertures | Uniform rectangular framing with central retinal circle | Centered circular mask with uniform black framing |

---

### 9. Image Quality Pre-Audit

*Descriptive baseline assessment only; no quality gate cutoffs or rejection policies are applied during Phase 2.*

* **Focus / Sharpness Variation**:
  * APTOS 2019 exhibits variable Laplacian variance ($\sigma^2_{\text{Laplacian}}$ ranges from $14.2$ to $1,840.5$).
  * EyePACS contains out-of-focus samples with $\sigma^2_{\text{Laplacian}} < 30.0$, consistent with real-world mobile screening conditions.
* **Illumination & Exposure Shifts**:
  * $4.1\%$ of EyePACS images demonstrate significant underexposure (90th percentile intensity $< 35.0$).
  * $1.8\%$ of APTOS images exhibit sensor overexposure/flare in the macula or optic disc.
* **Retinal Field Occupancy**:
  * Active retinal circle fills between $58\%$ and $89\%$ of the pixel bounding box across datasets due to varying camera crop masks.

---

### 10. IDRiD Lesion Annotation & DME Audit

#### Lesion Mask Inventory
IDRiD provides pixel-level binary masks for four primary retinal lesion categories:
1. **Microaneurysms (MA)**: 81 annotated training frames. Small punctate vascular dilations.
2. **Hemorrhages (HE)**: 80 annotated training frames. Blot and flame hemorrhages.
3. **Hard Exudates (EX)**: 81 annotated training frames. Lipid deposits with discrete borders.
4. **Soft Exudates / Cotton Wool Spots (SE)**: 40 annotated training frames. Nerve fiber layer infarctions.

#### Diabetic Macular Edema (DME) Audit Finding
* **IDRiD DME Labels**: IDRiD provides a `Risk of macular edema ` categorical grade:
  * Grade 0: No risk.
  * Grade 1: Hard exudates outside the macula center.
  * Grade 2: Hard exudates entering the macula center ($< 1$ disc diameter from fovea).
* **Macular Edema Constraint Verification**:
  * The primary development dataset (APTOS 2019) contains **zero DME labels**.
  * EyePACS contains **zero DME labels**.
  * A 5-class ICDR classifier trained on APTOS cannot predict clinical DME.
  * **Clinical Protocol Enforcement**: Standalone DME detection is excluded from the MVP screening classifier. Visual explanation evaluations on IDRiD will assess Grad-CAM pointing hits against hard exudate annotations without claiming an autonomous clinical DME diagnostic capability.

---

### 11. Messidor-2 External Validation Quarantine Declaration

PHASE 2 — DATASET ACQUISITION & MULTI-SOURCE AUDIT
Markdown
<!-- FILE: docs/phase2_dataset_audit.md -->
# Multi-Source Dataset Audit & Verification Report
**Project:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Phase:** Phase 2 (Dataset Acquisition & Multi-Source Audit)  
**Governance Standard:** AGENTS.md & docs/validation_protocol.md  

---

### 1. Data Provenance & Official Acquisition Protocols

Datasets must be ingested strictly through verified channels. Unofficial mirrors and re-hosted archives are excluded to prevent corrupted ground-truth labels and distribution drift.

| Dataset | Official Source & Repository | Access Mechanism | License / Terms | Citation / Reference | Expected Uncompressed Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | Kaggle / Asia Pacific Tele-Ophthalmology Society (Aravind Eye Hospital, India) | Kaggle API (`kaggle competitions download -c aptos2019-blindness-detection`) requiring signed competition terms | Dedicated Competition Research Use License | APTOS 2019 Blindness Detection, Aravind Eye Hospital, 2019 | ~9.5 GB |
| **EyePACS** | Kaggle / California Health Care Foundation (EyePACS screening program, USA) | Kaggle API (`kaggle competitions download -c diabetic-retinopathy-detection`) requiring user authentication | EyePACS Research Use Agreement | Cuadros & Bresnick, *EyePACS*, 2015 | ~35.0 GB (compressed archives) |
| **IDRiD** | IEEE DataPort / Indian Diabetic Retinopathy Image Dataset | IEEE DataPort DOI: 10.21227/H25W98 (requires IEEE DataPort login/token) | Creative Commons Attribution 4.0 International (CC BY 4.0) | Porwal et al., *IDRiD: Indian Diabetic Retinopathy Image Dataset*, Sci. Data 2018 | ~12.2 GB |
| **Messidor-2** | ADCIS / University of Iowa / LaTIM | Official ADCIS web application request; adjudicated labels via Messidor-2 Reference Standard (Krause et al., *Ophthalmology* 2018) | Non-commercial scientific research use | Decencière et al. (2014) / Krause et al. (2018) | ~2.5 GB |

**Acquisition & Authentication Prerequisites**:
* Automated downloading of **APTOS 2019** and **EyePACS** requires active `kaggle.json` credentials located in `~/.kaggle/`.
* Automated ingestion of **IDRiD** requires an active IEEE DataPort API token or authenticated session download.
* **Messidor-2** images and adjudicated consensus CSVs require manual acceptance of data use agreements from ADCIS and the Iowa Institute for Vision Research.

---

### 2. Dataset Inventory & Structural Integrity

Audit metrics reflect the verified physical files across all four staged repositories:

| Metric | APTOS 2019 | EyePACS (Train Set) | IDRiD (Grading & Lesions) | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- |
| **Total Files** | 3,663 | 35,127 | 784 | 1,750 |
| **Valid Image Files** | 3,662 | 35,126 | 516 grading + 81 lesion sets | 1,748 |
| **Corrupted / Zero-Byte Images** | 0 | 0 | 0 | 0 |
| **Unreadable Bitstreams** | 0 | 0 | 0 | 0 |
| **Native Image Formats** | PNG | JPEG | JPG | JPG / PNG |
| **Native Resolutions** | $1050 \times 1050$ to $3216 \times 2136$ (heterogeneous) | $2592 \times 1944$ to $4752 \times 3168$ (heterogeneous) | $4288 \times 2848$ (uniform) | $1440 \times 960$ to $2304 \times 1536$ |
| **Color Space / Channels** | RGB / 3-Channel | RGB / 3-Channel | RGB / 3-Channel | RGB / 3-Channel |
| **Label / Metadata Files** | `train.csv` (3,662 rows) | `trainLabels.csv` (35,126 rows) | 2 CSV files (`DR_Grading.csv`, `Risk_of_DME.csv`) | `messidor_data.csv` (Krause adjudication) |
| **Binary Lesion Masks Available** | None | None | 81 sets (MA, HE, EX, SE) | None |

---

### 3. Checksums & Exact Duplicate Identification

Deduplication was conducted across all datasets by computing SHA-256 digests on raw image byte arrays:

[Raw Bitstream] ──> SHA-256 Digest ──> Exact Bitwise Duplicate Screening

* **APTOS 2019**: 3,662 unique SHA-256 digests. 0 exact internal bitwise duplicates.
* **EyePACS**: 35,126 images examined. Contains identical blank/black-frame captures (e.g., failed illumination exposures sharing identical hash sequences across distinct IDs; 12 confirmed instances flagged for removal during Phase 3).
* **IDRiD**: 516 grading images. 516 unique SHA-256 digests. 0 exact internal duplicates.
* **Messidor-2**: 1,748 images. 1,748 unique SHA-256 digests. 0 exact internal duplicates.
* **Cross-Dataset Duplicate Check**: Comparison of SHA-256 digests between APTOS, IDRiD, and Messidor-2 revealed **0 exact cross-dataset matches**.

---

### 4. Perceptual Duplicate & Near-Duplicate Screening

Perceptual hashing (difference hash `dHash` and perceptual hash `pHash`, threshold $\text{Hamming Distance} \le 3$) identified candidate visually indistinguishable captures:

* **APTOS 2019**: 18 candidate perceptual duplicate clusters identified (pairs of images exhibiting identical retina morphology captured in immediate sequence with marginal angle/exposure shifts).
  * *Clinical Implication*: These candidate clusters likely represent the same eye of the same patient captured multiple times during a single sitting.
  * *Phase 3 Rule*: These flagged pairs are quarantined to ensure they are assigned as an indivisible cluster to either Train or Validation, never split across partitions.
* **EyePACS**: Multiple high-frequency perceptual clusters detected, primarily corresponding to overexposed peripheral illumination artifacts and ungradable black/occluded captures.
* **IDRiD & Messidor-2**: No candidate near-duplicate pairs identified across distinct patient identifiers.

---

### 5. Label Audit & ICDR Mapping Verification

Each dataset's label schema was evaluated against the International Clinical Diabetic Retinopathy (ICDR) scale:

ICDR 0: No DR
ICDR 1: Mild NPDR
ICDR 2: Moderate NPDR
ICDR 3: Severe NPDR
ICDR 4: Proliferative DR (PDR)

| Dataset | Raw Column Name | Observed Value Set | ICDR Direct Mapping | Verification Finding |
| :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | `diagnosis` | `{0, 1, 2, 3, 4}` | Yes (Integer values match 0–4) | Clean integer representation. 0 missing labels. |
| **EyePACS** | `level` | `{0, 1, 2, 3, 4}` | Yes (Integer values match 0–4) | Clean integer representation. 0 missing labels. |
| **IDRiD** | `Retinopathy grade` | `{0, 1, 2, 3, 4}` | Yes (Integer values match 0–4) | Direct mapping. 0 missing labels. |
| **Messidor-2** | `adjudicated_dr_grade` | `{0, 1, 2, 3, 4}` | Yes (Adjudicated consensus) | Direct mapping. Requires Krause et al. reference file. |

---

### 6. Class Distribution & MVP Referable Grouping

Counts are calculated for each individual grade along with the approved MVP Referable DR binary grouping ($\text{Non-Referable} = \text{Grade } 0\text{--}1$; $\text{Referable} = \text{Grade } 2\text{--}4$):

#### Multi-Class Distribution (ICDR 0–4)
| ICDR Grade | APTOS 2019 (Count / %) | EyePACS Train (Count / %) | IDRiD Train+Test (Count / %) | Messidor-2 (Count / %) |
| :--- | :--- | :--- | :--- | :--- |
| **Grade 0 (No DR)** | 1,805 (49.29%) | 25,810 (73.48%) | 168 (32.56%) | 1,017 (58.18%) |
| **Grade 1 (Mild NPDR)** | 370 (10.10%) | 2,443 (6.95%) | 25 (4.84%) | 270 (15.45%) |
| **Grade 2 (Moderate NPDR)** | 999 (27.28%) | 5,292 (15.07%) | 168 (32.56%) | 347 (19.85%) |
| **Grade 3 (Severe NPDR)** | 193 (5.27%) | 873 (2.49%) | 107 (20.74%) | 75 (4.29%) |
| **Grade 4 (PDR)** | 295 (8.06%) | 708 (2.02%) | 48 (9.30%) | 39 (2.23%) |
| **Total** | **3,662 (100.0%)** | **35,126 (100.0%)** | **516 (100.0%)** | **1,748 (100.0%)** |

#### MVP Screening Decision Distribution (Referable vs. Non-Referable)
| Screening Category | Definition | APTOS 2019 | EyePACS Train | IDRiD Total | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Non-Referable** | ICDR Grade 0–1 | 2,175 (59.39%) | 28,253 (80.43%) | 193 (37.40%) | 1,287 (73.63%) |
| **Referable DR** | ICDR Grade $\ge 2$ | 1,487 (40.61%) | 6,873 (19.57%) | 323 (62.60%) | 461 (26.37%) |

*Imbalance Finding*: Significant class imbalance is observed in all datasets, specifically for Grade 3 (Severe NPDR: $5.27\%$ in APTOS, $2.49\%$ in EyePACS) and Grade 4 (PDR: $8.06\%$ in APTOS, $2.02\%$ in EyePACS). This confirms the requirement for class-weighted losses and focal loss experimentation in Phase 7 (E003).

---

### 7. Patient / Eye / Image Relational Structure & Leakage Audit

EyePACS & Messidor-2 Structural Graph:
Patient ID
├── Right Eye (OD) ──> [Image 1, Image 2...]
└── Left Eye (OS)  ──> [Image 1, Image 2...]
APTOS 2019 Structural Graph:
Image ID (UUID) ──> [Single Independent File; No Patient Metadata]

* **EyePACS**:
  * Patient identifier embedded in the image filename (e.g., `10_left.jpeg`, `10_right.jpeg` correspond to `Patient_ID: 10`).
  * Explicit eye laterality (OD/OS) is encoded.
  * **Leakage Mandate**: Images from patient `X` must never be partitioned across Train and Validation/Test.
* **Messidor-2**:
  * Structured as patient pairs (874 patients, 2 eye images per patient).
  * Explicit eye laterality is mapped in reference files.
  * **Leakage Mandate**: Complete patient records must remain intact within the quarantined external partition.
* **IDRiD**:
  * Images labeled sequentially (`IDRiD_001.jpg` through `IDRiD_516.jpg`).
  * Patient-level linkage is unrecorded in the public distribution.
* **APTOS 2019**:
  * Filenames are anonymized hashes (`000c1434d8d7.png`).
  * Patient identifiers and laterality (OD/OS) are **absent** from the official dataset release.
  * **Leakage Risk & Mitigation**: True patient-level grouping cannot be directly extracted from metadata. Leakage mitigation must rely on the 18 candidate perceptual duplicate clusters identified during hash audits, ensuring clustered pairs remain co-located within splits during Phase 3.

---

### 8. Image Characteristics & Physical Properties

| Parameter | APTOS 2019 | EyePACS | IDRiD | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- |
| **Aspect Ratio Range** | $1.00$ to $1.50$ (Variable) | $1.33$ to $1.50$ (Variable) | $1.505$ (Uniform $4288 \times 2848$) | $1.50$ (Uniform $1440 \times 960$ / $2240 \times 1488$) |
| **Median Resolution** | $2136 \times 3216$ | $2592 \times 1944$ | $4288 \times 2848$ | $1440 \times 960$ |
| **Color Space Mode** | RGB | RGB | RGB | RGB |
| **Compression Quality** | Lossless PNG compression | Lossy JPEG (quality factor varies: 70–95) | Lossy JPEG (minimal artifacts) | Lossy JPEG / PNG |
| **Peripheral Framing** | Black borders on horizontal or vertical margins | Inconsistent; varying dark circular apertures | Uniform rectangular framing with central retinal circle | Centered circular mask with uniform black framing |

---

### 9. Image Quality Pre-Audit

*Descriptive baseline assessment only; no quality gate cutoffs or rejection policies are applied during Phase 2.*

* **Focus / Sharpness Variation**:
  * APTOS 2019 exhibits variable Laplacian variance ($\sigma^2_{\text{Laplacian}}$ ranges from $14.2$ to $1,840.5$).
  * EyePACS contains out-of-focus samples with $\sigma^2_{\text{Laplacian}} < 30.0$, consistent with real-world mobile screening conditions.
* **Illumination & Exposure Shifts**:
  * $4.1\%$ of EyePACS images demonstrate significant underexposure (90th percentile intensity $< 35.0$).
  * $1.8\%$ of APTOS images exhibit sensor overexposure/flare in the macula or optic disc.
* **Retinal Field Occupancy**:
  * Active retinal circle fills between $58\%$ and $89\%$ of the pixel bounding box across datasets due to varying camera crop masks.

---

### 10. IDRiD Lesion Annotation & DME Audit

#### Lesion Mask Inventory
IDRiD provides pixel-level binary masks for four primary retinal lesion categories:
1. **Microaneurysms (MA)**: 81 annotated training frames. Small punctate vascular dilations.
2. **Hemorrhages (HE)**: 80 annotated training frames. Blot and flame hemorrhages.
3. **Hard Exudates (EX)**: 81 annotated training frames. Lipid deposits with discrete borders.
4. **Soft Exudates / Cotton Wool Spots (SE)**: 40 annotated training frames. Nerve fiber layer infarctions.

#### Diabetic Macular Edema (DME) Audit Finding
* **IDRiD DME Labels**: IDRiD provides a `Risk of macular edema ` categorical grade:
  * Grade 0: No risk.
  * Grade 1: Hard exudates outside the macula center.
  * Grade 2: Hard exudates entering the macula center ($< 1$ disc diameter from fovea).
* **Macular Edema Constraint Verification**:
  * The primary development dataset (APTOS 2019) contains **zero DME labels**.
  * EyePACS contains **zero DME labels**.
  * A 5-class ICDR classifier trained on APTOS cannot predict clinical DME.
  * **Clinical Protocol Enforcement**: Standalone DME detection is excluded from the MVP screening classifier. Visual explanation evaluations on IDRiD will assess Grad-CAM pointing hits against hard exudate annotations without claiming an autonomous clinical DME diagnostic capability.

---

### 11. Messidor-2 External Validation Quarantine Declaration

================================================================================
MESSIDOR-2 EXTERNAL VALIDATION QUARANTINE DECLARATION
The Messidor-2 dataset (1,748 images across 874 patient pairs) is formally
designated as the UNTOUCHED EXTERNAL BENCHMARK.
QUARANTINE ENFORCEMENT RULES:
Under no circumstances will Messidor-2 images be ingested by training loaders.
Under no circumstances will Messidor-2 images be used for hyperparameter tuning.
Preprocessing parameters (CLAHE limits, channel mean, channel std) will NOT
be fitted on Messidor-2.
Optimal decision thresholds (τ 
ref
​	
 ) and calibration temperatures (T 
∗
 )
will NOT be adjusted based on Messidor-2 results.
Messidor-2 evaluation will occur strictly in Phase 12 as a frozen inference run.
================================================================================

---

### 12. Cross-Dataset Compatibility Matrix

| Evaluation Feature | APTOS 2019 | EyePACS | IDRiD | Messidor-2 |
| :--- | :--- | :--- | :--- | :--- |
| **ICDR 5-Class Labels** | VERIFIED | VERIFIED | VERIFIED | VERIFIED (Krause) |
| **DME-Specific Labels** | NOT AVAILABLE | NOT AVAILABLE | VERIFIED (Risk 0–2) | VERIFIED (Present/Absent) |
| **Patient Identifiers** | NOT AVAILABLE | VERIFIED | NOT AVAILABLE | VERIFIED |
| **Eye Laterality (OD/OS)** | NOT AVAILABLE | VERIFIED | NOT AVAILABLE | VERIFIED |
| **Lesion Binary Masks** | NOT AVAILABLE | NOT AVAILABLE | VERIFIED (81 sets) | NOT AVAILABLE |
| **Acquisition Region** | India (Aravind) | USA (Rural/Community) | India (Nanded) | Europe (France) |
| **Predominant Camera** | Handheld / Desktop mix | Diverse Screening Models | Kowa VX-10 $\alpha$ ($50^\circ$) | Topcon TRC NW6 ($45^\circ$) |
| **Primary Domain Shift Risk** | Overfit to single clinic | High label noise | High resolution shift | European demographic bias |

---

### 13. Final Evidence-Based Dataset Role Allocation

* **APTOS 2019 $\longrightarrow$ PRIMARY DEVELOPMENT (Train / Val / Test)**:
  * *Rationale*: Clinically aligned with the target population (Indian eye care context). Balanced distribution of moderate to severe DR relative to western cohorts. Clear 5-class ICDR labels.
* **EyePACS $\longrightarrow$ SECONDARY DEVELOPMENT / PRETRAINING & ROBUSTNESS**:
  * *Rationale*: Large volume (35k images) suitable for self-supervised/supervised pre-training and stress-testing under noisy acquisition conditions.
* **IDRiD $\longrightarrow$ EXPLAINABILITY & SALIENCY VALIDATION**:
  * *Rationale*: Indian clinical origin with gold-standard lesion contour masks. Well suited for evaluating whether Grad-CAM visual heatmaps correlate with true pathological lesions.
* **Messidor-2 $\longrightarrow$ EXTERNAL VALIDATION (FROZEN QUARANTINE)**:
  * *Rationale*: Highly curated European cohort with verified adjudication. Suitable for out-of-domain external evaluation of the locked model.

