# Experiment E002: Preprocessing Comparison (CLAHE vs. Baseline)

### 1. Hypothesis & Objective
* **Hypothesis**: Contrast Limited Adaptive Histogram Equalization (CLAHE; clip limit = 2.0, tile grid = 8x8) on the Lightness channel of retinal fundus images enhances faint vascular structures and microaneurysms, thereby improving validation Macro F1 score compared to the unenhanced E001 baseline.
* **Controlled Comparison**:
  * **Control (E001)**: Retinal crop -> Bilinear resize to 512x512 -> [0, 1] float32 normalization.
  * **Experiment (E002)**: Retinal crop -> Bilinear resize to 512x512 -> CLAHE -> [0, 1] float32 normalization.
* **Unchanged Controlled Variables**:
  * Backbone: Pretrained ImageNet EfficientNet-B0
  * Optimizer: AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-5}$)
  * Loss: Standard CrossEntropyLoss (unweighted)
  * Batch size: 16, Maximum epochs: 30, Early stopping patience: 7
  * Random seed: 42, Compute device: MPS (Apple M4)
  * Splits: Train (2,563 images), Validation (549 images).
* **Test Set Policy**: The held-out test split (`test.csv`) was **NOT evaluated** to avoid validation leakage and preserve test set integrity.

---

### 2. Validation Results & Comparison Against Baseline

| Metric | E001 Baseline (Validation) | E002 CLAHE (Validation) | Delta ($\text{CLAHE} - \text{Baseline}$) |
| :--- | :--- | :--- | :--- |
| **Best Epoch** | Epoch 18 | Epoch 16 | -2 epochs |
| **Accuracy** | **82.33%** | 81.24% | -1.09% |
| **Macro F1 (Primary Metric)**| **0.6712** | 0.6587 | **-0.0125** |
| **Quadratic Weighted Kappa (QWK)** | **0.8641** | 0.8504 | -0.0137 |
| **Referable Sensitivity** | 87.78% | **88.46%** | +0.68% |
| **Referable Specificity** | **93.35%** | 91.48% | -1.87% |

---

### 3. Per-Class Validation Performance (E002 CLAHE)

| Class ID | ICDR Stage Title | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | No DR | 0.9118 | 0.9058 | 0.9088 | 308 |
| **1** | Mild NPDR | 0.4643 | 0.4643 | 0.4643 | 56 |
| **2** | Moderate NPDR | 0.7452 | 0.7800 | 0.7622 | 150 |
| **3** | Severe NPDR | 0.5357 | 0.5172 | 0.5263 | 29 |
| **4** | Proliferative DR (PDR) | 0.7600 | 0.6552 | 0.7037 | 29 |

#### E002 Validation Confusion Matrix (Counts)

Pred 0   Pred 1   Pred 2   Pred 3   Pred 4
True 0        279       16       13        0        0
True 1         16       26       14        0        0
True 2         10       12      117        8        3
True 3          0        1       10       15        3
True 4          1        1        3        5       19

---

### 4. Decision & Conclusion

* **Decision**: **CLAHE IS REJECTED. E001 BASELINE IS RETAINED.**
* **Scientific Rationale**:
  1. Validation Macro F1 dropped by **-0.0125** (from 0.6712 to 0.6587). Under the pre-established decision rule, CLAHE failed to improve the primary model selection metric.
  2. Specificity decreased by **-1.87%** (from 93.35% to 91.48%), with Grade 0 false positives increasing (13 Grade 0 samples misclassified as Moderate NPDR, compared to 11 in baseline).
  3. Visual and error analyses confirm that CLAHE amplifies choroidal background textures and peripheral camera sensor noise, generating false microaneurysm patterns in healthy retinas. While referable sensitivity showed a slight increase (+0.68%), this does not justify the degradation in Macro F1, specificity, and QWK.
* **Limitations**:
  * Evaluated on a single fixed clip limit (2.0) and tile grid size (8, 8). Finer parameter grids or channel-specific contrast stretching could yield different noise characteristics, but under the controlled E002 protocol, standard CLAHE is inferior to the unenhanced baseline.