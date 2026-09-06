# Explainability & Clinical Evidence Validation (E008)

**Target Model**: `models/checkpoints/E007_best_model.pt` (Active Champion)  
**Methodology**: Gradient-weighted Class Activation Mapping (Grad-CAM) targeting EfficientNet-B0 `model.features[-1]`  
**Cohort Evaluated**: 10 Representative Cases across all 5 DR grades from `data/processed/aptos/validation.csv` ($N=601$)  

---

## 1. Saliency Attribution vs. Lesion Segmentation
Grad-CAM computes relative spatial attributions indicating which receptive fields contributed most strongly to logits. It represents **saliency localization**, **NOT** lesion segmentation or direct lesion detection.

## 2. Retinal Tissue vs. Border Artifact Analysis
Quantitative attention energy distribution across the 10 audit cases:
- **Mean Retinal Tissue Energy Fraction**: **83.99%**
- **Mean Non-Retinal / Border Artifact Energy**: **16.01%**
- **Diagnostic Finding**: Grad-CAM attention is concentrated within the vascular arcades and macula without corner clustering. Unlike E006, the remaining peripheral circular border energy does not induce class drift because E007 relies on categorical Softmax.

## 3. IDRiD Lesion Correspondence
Pixel-level lesion segmentation masks are not present in the active repository workspace. Consequently, IoU and Pointing Game metrics against ground-truth lesion masks were not evaluated.
