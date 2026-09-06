# Experiment E004: Training Augmentation (Geometric & Photometric)

### 1. Specification & Hypothesis
* **Hypothesis**: Training with clinically plausible, conservative retinal augmentations (horizontal reflection, small rotations $\le 10^\circ$, minor affine shifts, and mild brightness/contrast adjustments) prevents spatial overfitting and improves validation Macro F1, QWK, and referable sensitivity over the non-augmented E003 baseline.
* **Controlled Comparison**:
  * **Baseline (E003)**: No training augmentation (Crop -> Resize $512 \times 512$ -> $[0, 1]$ Normalization).
  * **Experiment (E004)**: Conservative training augmentation pipeline applied to training split.
* **Exact Augmentation Pipeline (Training Only)**:
  1. Horizontal Flip ($p = 0.5$)
  2. Random Rotation ($\pm 10^\circ$, $p = 0.7$)
  3. Random Affine Scaling ($[0.96, 1.04]$) and Translation ($\pm 2\%$, $p = 0.5$)
  4. Mild Color Jitter (Brightness $[0.95, 1.05]$, Contrast $[0.95, 1.05]$, $p = 0.5$)
* **Unchanged Controlled Variables**:
  * Backbone: EfficientNet-B0 (ImageNet pretrained, dropout 0.2, fine-tune full backbone)
  * Loss: Weighted Cross-Entropy ($w = [0.4059, 1.9810, 0.7330, 3.8048, 2.4711]$ from `train.csv`)
  * Optimizer: AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-5}$)
  * Scheduler: CosineAnnealingLR (30 epochs)
  * Batch Size: 16, Seed: 42, Compute Device: MPS (Apple M4)
  * Early Stopping: Patience = 7 on validation Macro F1
* **Test Set Policy**: `data/processed/aptos/test.csv` was **NOT loaded and NOT evaluated**.

---

### 2. Validation Results & Comparison Against E003 Baseline

| Metric | E003 Weighted-CE (Validation) | E004 Augmentation (Validation) | Delta ($\text{E004} - \text{E003}$) |
| :--- | :--- | :--- | :--- |
| **Best Epoch** | Epoch 21 | Epoch 22 | +1 epoch |
| **Accuracy** | 84.15% | **85.31%** | **+1.16%** |
| **Macro F1 (Primary Metric)**| 0.7096 | **0.7581** | **+0.0485** |
| **Quadratic Weighted Kappa (QWK)** | 0.8733 | **0.8876** | **+0.0143** |
| **Referable Sensitivity (Grade $\ge 2$)** | 90.87% | **94.23%** | **+3.36%** (Target: $\ge 90\%$) |
| **Referable Specificity (Grade $< 2$)** | 91.76% | **94.23%** | **+2.47%** (Target: $\ge 85\%$) |

---

### 3. Per-Class Validation Breakdown (E004 vs. E003)

| Class | Stage Title | E003 Val Recall | E004 Val Recall | E003 Val F1 | E004 Val F1 | Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | No DR | 88.31% (272/308) | **92.21%** (284/308) | 0.9051 | **0.9404** | 308 |
| **1** | Mild NPDR | 58.93% (33/56) | **66.07%** (37/56) | 0.5593 | **0.6435** | 56 |
| **2** | Moderate NPDR | 79.33% (119/150) | **84.00%** (126/150) | 0.7628 | **0.8235** | 150 |
| **3** | Severe NPDR | 58.62% (17/29) | **62.07%** (18/29) | 0.5574 | **0.5902** | 29 |
| **4** | Proliferative DR | 72.41% (21/29) | **79.31%** (23/29) | 0.7636 | **0.7931** | 29 |

#### E004 Validation Confusion Matrix

Pred 0   Pred 1   Pred 2   Pred 3   Pred 4
True 0        284       14       10        0        0
True 1          8       37       11        0        0
True 2          4        7      126       10        3
True 3          0        1        7       18        3
True 4          0        0        2        4       23

---

### 4. Decision & Conclusion

* **Decision**: **CONSERVATIVE TRAINING AUGMENTATION IS RETAINED.**
* **Scientific Rationale**:
  1. **Primary Metric**: Validation Macro F1 improved by **+0.0485** (from 0.7096 to 0.7581).
  2. **Uniform Disease Stage Gains**: F1-scores improved across all 5 ICDR classes:
     * Grade 0: $0.9051 \to 0.9404$ (+0.0353)
     * Grade 1: $0.5593 \to 0.6435$ (+0.0842)
     * Grade 2: $0.7628 \to 0.8235$ (+0.0607)
     * Grade 3: $0.5574 \to 0.5902$ (+0.0328)
     * Grade 4: $0.7636 \to 0.7931$ (+0.0295)
  3. **Clinical Screening Compliance**: Referable sensitivity reached **94.23%** (exceeding the $\ge 90\%$ clinical target), and referable specificity reached **94.23%** (exceeding the $\ge 85\%$ target).
  4. **Ordinal Consistency**: Quadratic Weighted Kappa improved to **0.8876** (+0.0143), confirming a reduction in distant ordinal misclassifications.
* **Limitations**:
  * Augmentation parameters were evaluated as an integrated conservative policy. Future ablation could explore isolated contributions of rotation vs. color jitter.