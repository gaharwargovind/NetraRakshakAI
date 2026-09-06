# Experiment E001: Baseline EfficientNet-B0 Classifier

### 1. Specification & Objectives
* **Experiment ID**: E001
* **Objective**: Establish the official unweighted, non-augmented 5-class ICDR baseline classifier using ImageNet pretrained EfficientNet-B0.
* **Dataset**: APTOS 2019 Blindness Detection (Split Phase 3: Train=2,563, Val=549, Test=550; quarantined duplicates excluded).
* **Backbone**: EfficientNet-B0 with custom head (`Dropout(p=0.2)` -> `Linear(1280, 5)`).
* **Optimization**: AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-5}$), $\text{CosineAnnealingLR}$ over 30 epochs.
* **Loss**: Standard CrossEntropyLoss (no class weights, no focal term).

---

### 2. Measured Results Summary

| Split / Benchmark | Accuracy | Macro F1 | Quadratic Weighted Kappa (QWK) | Referable Sensitivity (Grade $\ge 2$) | Referable Specificity (Grade $< 2$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation (Best Epoch 18)** | 82.33% | 0.6712 | 0.8641 | 87.78% | 93.35% |
| **Held-Out Test (Evaluated Once)** | 81.64% | 0.6628 | 0.8573 | 86.88% | 92.81% |

---

### 3. Clinical Target Analysis
* **Referable Sensitivity Target ($\ge 90.0\%$)**: **NOT ACHIEVED** (Held-out Test: $86.88\%$).
  * *Clinical Analysis*: The unweighted cross-entropy loss underperforms on minority Grade 3 (Severe NPDR: recall $48.28\%$) and Grade 1 (Mild: recall $52.73\%$), pulling referable sensitivity below the required clinical safety threshold.
* **Referable Specificity Target ($\ge 85.0\%$)**: **ACHIEVED** (Held-out Test: $92.81\%$).
* **Engineering Takeaway**: The baseline confirms the necessity of Phase 7 experiments:
  * **E003**: Class imbalance mitigation (Weighted Cross-Entropy and Focal Loss) to address minority class miss rates.
  * **E004**: Medically plausible data augmentations to improve feature invariance.