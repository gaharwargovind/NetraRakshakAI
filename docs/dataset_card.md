# Multi-Dataset Governance & Provenance Card
**Document Version:** 2.0 (Updated post-Phase 2 multi-source audit)  
**Project:** SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India  

---

### 1. APTOS 2019 Blindness Detection Dataset
* **Role**: Primary Model Development (Train / Validation / Internal Test Splits).
* **Clinical Provenance**: Aravind Eye Hospital, Tamil Nadu, India.
* **Sample Count**: 3,662 validated color fundus photographs.
* **Format**: Portable Network Graphics (PNG), variable resolutions ($1050 \times 1050$ to $3216 \times 2136$).
* **Annotation Schema**: Multi-class ICDR Grade 0 (1,805), 1 (370), 2 (999), 3 (193), 4 (295).
* **MVP Screening Distribution**: Non-Referable: 2,175 (59.39%); Referable DR: 1,487 (40.61%).
* **Known Limitations**: Anonymized file hashes; patient identifiers and laterality are absent. 18 candidate near-duplicate capture clusters identified via perceptual hashing.
* **Quarantine / Leakage Policy**: Candidate duplicate clusters are flagged to ensure strict co-location within partitions during Phase 3.

---

### 2. EyePACS Diabetic Retinopathy Dataset
* **Role**: Secondary Pretraining, Transfer Learning & Realistic Degradation Source.
* **Clinical Provenance**: Community and rural screening clinics across the United States.
* **Sample Count**: 35,126 training images evaluated.
* **Format**: JPEG bitstreams, resolutions spanning $2592 \times 1944$ to $4752 \times 3168$.
* **Annotation Schema**: 5-class DR scale (0: 25,810; 1: 2,443; 2: 5,292; 3: 873; 4: 708).
* **Relational Metadata**: Full patient-level identifiers and laterality encoded directly in filenames (`{patient_id}_{eye}.jpeg`).
* **Known Limitations**: Observable label noise resulting from multi-reader variance; presence of completely occluded/underexposed captures requiring quality interception.

---

### 3. Indian Diabetic Retinopathy Image Dataset (IDRiD)
* **Role**: Visual Explainability (Grad-CAM) Pointing Evaluation.
* **Clinical Provenance**: Eye Clinic in Nanded, Maharashtra, India.
* **Sample Count**: 516 clinical grading photographs; 81 pixel-annotated lesion subsets.
* **Format**: Uniform high-resolution JPEG ($4288 \times 2848$, $50^\circ$ Field of View).
* **Annotations Available**: 5-class ICDR grade, 3-class DME risk scale, binary segmentation contours for microaneurysms, hemorrhages, hard exudates, and soft exudates.
* **Role Boundary**: Excluded from core classification training. Lesion contours will be used in Phase 9 to evaluate visual attention alignment.

---

### 4. Messidor-2 Reference Standard
* **Role**: Quarantined External Validation Benchmark (Evaluated frozen in Phase 12).
* **Clinical Provenance**: Three French ophthalmologic clinical centers.
* **Sample Count**: 1,748 images (874 bilateral patient pairs).
* **Format**: Color fundus images ($1440 \times 960$ to $2304 \times 1536$).
* **Adjudicated Labels**: ICDR consensus grades (0: 1,017; 1: 270; 2: 347; 3: 75; 4: 39).
* **Screening Distribution**: Non-Referable: 1,287 (73.63%); Referable DR: 461 (26.37%).
* **Quarantine Mandate**: No images, labels, or distribution parameters from Messidor-2 will enter model training, validation tuning, threshold selection, or calibration.