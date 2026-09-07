# External Dataset Card: Indian Diabetic Retinopathy Image Dataset (IDRiD)

## 1. Dataset Provenance & Overview
- **Dataset Name:** Indian Diabetic Retinopathy Image Dataset (IDRiD)
- **Official Source:** IEEE DataPort / ISBI 2018 Diabetic Retinopathy Challenge
- **Clinical Setting:** Eye Clinic in Nanded, Maharashtra, India
- **Acquisition Hardware:** Kowa VX-10alpha digital fundus camera (50-degree Field of View)
- **Native Resolution:** 4288 x 2848 pixels (Lossless/High-quality JPEG)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

## 2. Cohort Structure (Disease Grading Sub-Challenge)
- **Total Images:** 516 fundus examinations
  - Sub-split A (Designated Training): 413 images (`IDRiD_001.jpg` - `IDRiD_413.jpg`)
  - Sub-split B (Designated Testing): 103 images (`IDRiD_414.jpg` - `IDRiD_516.jpg`)
- **Combined External Cohort:** All 516 verified images will be evaluated as a single, pooled external cohort by the frozen E007 model.

## 3. Ground Truth Grading Semantics
Retinopathy severity was graded according to the International Clinical Diabetic Retinopathy (ICDR) scale:
- **Grade 0 (No DR):** No microaneurysms or retinal hemorrhages.
- **Grade 1 (Mild NPDR):** Microaneurysms only.
- **Grade 2 (Moderate NPDR):** More than microaneurysms, but less than severe.
- **Grade 3 (Severe NPDR):** 4-2-1 rule satisfied without signs of proliferation.
- **Grade 4 (PDR):** Neovascularization, vitreous/preretinal hemorrhage.

## 4. Known Differences from APTOS 2019
- **Camera Optics:** Kowa VX-10alpha (50-degree FOV) vs. mixed camera models in APTOS.
- **Illumination & Pigmentation:** Both cohorts capture South Asian Indian retinal phenotypes, providing a controlled test of optical/camera shift without demographic confounding.
- **Resolution:** Native resolution (4288 x 2848) is substantially higher than typical APTOS images, testing the downsampling and circular crop pipeline.
