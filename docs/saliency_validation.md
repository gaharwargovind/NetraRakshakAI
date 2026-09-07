# Phase E014: Counterfactual & Visual Saliency Validation

## 1. Executive Summary & Epistemic Scope
Phase E014 investigates whether the frozen E007 model relies on physiologically grounded visual evidence versus potentially spurious background features. 

> **Scientific Attribution Mandate:** Grad-CAM provides attribution evidence regarding image regions influencing the model output. Attribution localization must be strictly distinguished from lesion segmentation, counterfactual causality, and clinical diagnostic validity.

---

## 2. Step 0 Data Availability Audit Finding
Inspection of `data/raw/idrid/` confirmed that raw fundus photographs ($N=455$) and disease grades are present, but **lesion-level segmentation masks (microaneurysms, hemorrhages, exudates, optic disc) are completely absent**.
In accordance with governance:
* Quantitative lesion-overlap calculations (IoU, Dice, Pointing Game) were **suspended**.
* Formal status: *"Lesion-level quantitative validation unavailable for this cohort."*

---

## 3. Case Selection Protocol (Protocols A through J)
To eliminate manual cherry-picking, cases were deterministically selected from the validation cohorts:
* **Protocol A (Correct Grade 0):** Max-confidence correct normal eye.
* **Protocol B (Correct Grade 1):** Max-confidence correct Mild NPDR.
* **Protocol C (Correct Grade 2):** Max-confidence correct Moderate NPDR.
* **Protocol D (Correct Grade 3):** Max-confidence correct Severe NPDR.
* **Protocol E (Correct Grade 4):** Max-confidence correct Proliferative DR.
* **Protocol F (False Negative Referable):** True $\ge 2$, Predicted $< 2$.
* **Protocol G (False Positive Referable):** True $< 2$, Predicted $\ge 2$.
* **Protocol H (High-Confidence Error):** Incorrect prediction with highest softmax probability.
* **Protocol I (Low-Confidence Prediction):** Minimum top-1 vs. top-2 probability margin.
* **Protocol J (External IDRiD Domain-Shift Failure):** True Grade 2 under-referred on IDRiD.

---

## 4. Counterfactual Intervention Framework
To test whether salient regions causally drive predictions:
1. **Salient-Region Ablation ($CAM \ge 0.50$):** High-attribution pixels are occluded (zeroed). A causal model will exhibit a sharp decrease in predicted grade and referable probability.
2. **Background-Region Ablation ($CAM < 0.50$):** Non-salient background pixels are occluded. If the model is lesion-grounded, it will preserve the original diagnostic grade.

---

## 5. Spurious Correlation & Peripheral Leakage
Saliency energy is integrated across the segmented retinal aperture:
$$	ext{Leakage Ratio} = rac{\sum_{x 
otin \Omega_M} 	ext{CAM}(x)}{\sum_{x \in \Omega} 	ext{CAM}(x)}$$
Evaluates whether optical borders, vignetting, or camera illumination gradients falsely attract model attention.
