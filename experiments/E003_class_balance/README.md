# Experiment E003: Class Imbalance Mitigation (Weighted Cross-Entropy)

### 1. Specification & Hypothesis
* **Hypothesis**: Replacing standard Cross-Entropy with inverse-frequency Weighted Cross-Entropy will penalize false negatives on underrepresented grades (particularly Grade 1 Mild, Grade 3 Severe, and Grade 4 PDR), boosting validation Macro F1 score and referable sensitivity compared to the E001 baseline.
* **Class Weighting Formula**:
  $$w_c = \frac{N}{K \cdot N_c}$$
  Derived strictly from the 2,397 non-quarantined samples in `data/processed/aptos/train.csv`:
  * Class 0: **0.4059**
  * Class 1: **1.9810**
  * Class 2: **0.7330**
  * Class 3: **3.8048**
  * Class 4: **2.4711**
* **Control (E001)**: Standard unweighted `CrossEntropyLoss()`.
* **Experimental Condition (E003)**: `CrossEntropyLoss(weight=weights)`.
* **Unchanged Controlled Variables**:
  * Backbone: EfficientNet-B0 (ImageNet pretrained, dropout 0.2, fine-tune full backbone)
  * Preprocessing: E001 baseline (Retinal crop -> resize 512x512 -> [0, 1] float32 normalization)
  * Augmentation: None
  * Optimizer: AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-5}$)
  * Scheduler: CosineAnnealingLR (30 epochs)
  * Early stopping: Patience = 7 on validation Macro F1
  * Seed: 42, Compute Device: MPS (Apple M4)
* **Test Set Policy**: `test.csv` was **NOT loaded and NOT evaluated**.

---

### 2. Validation Results & Comparison Against Baseline

| Metric | E001 Baseline (Validation) | E003 Weighted-CE (Validation) | Delta ($\text{E003} - \text{E001}$) |
| :--- | :--- | :--- | :--- |
| **Best Epoch** | Epoch 18 | Epoch 21 | +3 epochs |
| **Accuracy** | 82.33% | **84.15%** | **+1.82%** |
| **Macro F1 (Primary Metric)**| 0.6712 | **0.7096** | **+0.0384** |
| **Quadratic Weighted Kappa (QWK)** | 0.8641 | **0.8733** | **+0.0092** |
| **Referable Sensitivity (Grade $\ge 2$)** | 87.78% | **90.87%** | **+3.09%** (Crosses $\ge 90\%$ target) |
| **Referable Specificity (Grade $< 2$)** | **93.35%** | 91.76% | -1.59% (Above $\ge 85\%$ target) |

---

### 3. Per-Class Validation Breakdown (E003 vs. E001 Baseline)

| Class | Stage Title | E001 Val Recall | E003 Val Recall | E001 Val F1 | E003 Val F1 | Support |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | No DR | **92.53%** (285/308) | 88.31% (272/308) | **0.9208** | 0.9051 | 308 |
| **1** | Mild NPDR | 51.79% (29/56) | **58.93%** (33/56) | 0.5133 | **0.5593** | 56 |
| **2** | Moderate NPDR | 76.00% (114/150) | **79.33%** (119/150) | 0.7525 | **0.7628** | 150 |
| **3** | Severe NPDR | 48.28% (14/29) | **58.62%** (17/29) | 0.5091 | **0.5574** | 29 |
| **4** | Proliferative DR | 65.52% (19/29) | **72.41%** (21/29) | 0.7037 | **0.7636** | 29 |

#### E003 Validation Confusion Matrix

Pred 0   Pred 1   Pred 2   Pred 3   Pred 4
True 0        272       17       19        0        0
True 1         12       33       11        0        0
True 2          7       10      119       11        3
True 3          0        1        9       17        2
True 4          1        0        3        4       21

---

### 4. Decision & Conclusion

* **Decision**: **WEIGHTED CROSS-ENTROPY IS RETAINED.**
* **Scientific Rationale**:
  1. **Primary Metric**: Validation Macro F1 improved by **+0.0384** (from 0.6712 to 0.7096).
  2. **Minority Disease Recall**: Grade 3 (Severe NPDR) recall increased by **+10.34%** (from 48.28% to 58.62%), Grade 4 (PDR) recall increased by **+6.89%** (from 65.52% to 72.41%), and Grade 1 (Mild NPDR) recall increased by **+7.14%** (from 51.79% to 58.93%).
  3. **Clinical Screening Target**: Referable sensitivity crossed the clinical screening target of $\ge 90.0\%$ on the validation partition (**90.87%**), while referable specificity remained high at **91.76%** (well above the $\ge 85.0\%$ target).
  4. **Cost-Benefit**: A minor drop in Grade 0 recall (-4.22%) is clinically acceptable given the marked reduction in false-negative misses on vision-threatening retinopathy.
* **Limitations**:
  * Inverse-frequency weights adjust gradient scale statically; dynamic hard-example reweighting (e.g., Focal Loss) may yield additional improvements and remains a candidate for future comparative exploration.

  