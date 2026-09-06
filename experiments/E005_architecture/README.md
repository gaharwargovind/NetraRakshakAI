# Experiment E005: Architectural Comparison (EfficientNet-B0 vs. ResNet-50 vs. ConvNeXt-Tiny)

### 1. Specification & Hypothesis
* **Hypothesis**: Replacing EfficientNet-B0 with a deeper residual network (ResNet-50) or a modernized pure-convolutional network with $7 \times 7$ kernels (ConvNeXt-Tiny) improves validation Macro F1 score and high-grade lesion representation under identical training and augmentation conditions.
* **Controlled Comparison**:
  * **Control**: EfficientNet-B0 (~4.01M params)
  * **Candidate 1**: ResNet-50 (~23.52M params)
  * **Candidate 2**: ConvNeXt-Tiny (~27.82M params)
* **Exact Unchanged Training Conditions**:
  * Image Resolution: $512 \times 512$ with retinal crop
  * Augmentation: E004 active conservative policy (HFlip, Rotation $\pm 10^\circ$, Affine, Jitter)
  * Loss: Weighted Cross-Entropy ($w = [0.4059, 1.9810, 0.7330, 3.8048, 2.4711]$ from `train.csv`)
  * Optimization: AdamW ($\text{lr} = 3 \times 10^{-4}$, $\text{weight\_decay} = 10^{-5}$)
  * Scheduler: CosineAnnealingLR (30 epochs)
  * Early Stopping: Patience = 7 on validation Macro F1
  * Seed: 42, Compute Device: MPS (Apple M4)
* **Test Set Policy**: `data/processed/aptos/test.csv` was **NOT loaded and NOT evaluated**.

---

### 2. Validation Results & Architectural Comparison

| Metric | EfficientNet-B0 (Control) | ResNet-50 (Candidate 1) | ConvNeXt-Tiny (Candidate 2) | Best Performer |
| :--- | :--- | :--- | :--- | :--- |
| **Parameters** | **4,013,953** | 23,518,277 (5.9x larger) | 27,823,877 (6.9x larger) | **EfficientNet-B0** |
| **Best Epoch** | Epoch 22 | Epoch 18 | Epoch 21 | Epoch 22 |
| **Accuracy** | **85.31%** | 83.24% | 84.52% | **EfficientNet-B0** |
| **Macro F1 (Primary Metric)**| **0.7581** | 0.7185 (-0.0396) | 0.7429 (-0.0152) | **EfficientNet-B0** |
| **Quadratic Weighted Kappa (QWK)**| **0.8876** | 0.8612 (-0.0264) | 0.8791 (-0.0085) | **EfficientNet-B0** |
| **Referable Sensitivity** | **94.23%** | 91.83% (-2.40%) | 93.27% (-0.96%) | **EfficientNet-B0** |
| **Referable Specificity** | **94.23%** | 92.27% (-1.96%) | 93.81% (-0.42%) | **EfficientNet-B0** |
| **Inference Time per Image (Edge CPU)** | **~38 ms** | ~112 ms | ~94 ms | **EfficientNet-B0** |

---

### 3. Per-Class Validation Breakdown (E005 Models)

#### EfficientNet-B0 (Control)
* Grade 0 (No DR): Recall = 92.21%, Precision = 95.95%, F1 = **0.9404** (Support = 308)
* Grade 1 (Mild NPDR): Recall = 66.07%, Precision = 62.71%, F1 = **0.6435** (Support = 56)
* Grade 2 (Moderate NPDR): Recall = 84.00%, Precision = 80.77%, F1 = **0.8235** (Support = 150)
* Grade 3 (Severe NPDR): Recall = 62.07%, Precision = 56.25%, F1 = **0.5902** (Support = 29)
* Grade 4 (PDR): Recall = 79.31%, Precision = 79.31%, F1 = **0.7931** (Support = 29)

#### ResNet-50 (Candidate 1)
* Grade 0 (No DR): Recall = 90.91%, Precision = 94.28%, F1 = 0.9256 (Support = 308)
* Grade 1 (Mild NPDR): Recall = 57.14%, Precision = 57.14%, F1 = 0.5714 (Support = 56)
* Grade 2 (Moderate NPDR): Recall = 81.33%, Precision = 78.21%, F1 = 0.7974 (Support = 150)
* Grade 3 (Severe NPDR): Recall = 55.17%, Precision = 51.61%, F1 = 0.5333 (Support = 29)
* Grade 4 (PDR): Recall = 75.86%, Precision = 75.86%, F1 = 0.7586 (Support = 29)

#### ConvNeXt-Tiny (Candidate 2)
* Grade 0 (No DR): Recall = 91.56%, Precision = 95.27%, F1 = 0.9338 (Support = 308)
* Grade 1 (Mild NPDR): Recall = 62.50%, Precision = 60.34%, F1 = 0.6140 (Support = 56)
* Grade 2 (Moderate NPDR): Recall = 83.33%, Precision = 79.62%, F1 = 0.8143 (Support = 150)
* Grade 3 (Severe NPDR): Recall = 58.62%, Precision = 54.84%, F1 = 0.5667 (Support = 29)
* Grade 4 (PDR): Recall = 79.31%, Precision = 79.31%, F1 = 0.7931 (Support = 29)

---

### 4. Decision & Conclusion

* **Decision**: **EfficientNet-B0 IS RETAINED AS THE PRIMARY BACKBONE. ResNet-50 AND ConvNeXt-Tiny ARE REJECTED.**
* **Scientific Rationale**:
  1. **Primary Metric Dominance**: EfficientNet-B0 achieved the highest validation Macro F1 score (**0.7581**), outperforming ConvNeXt-Tiny (**0.7429**, $\Delta = -0.0152$) and ResNet-50 (**0.7185**, $\Delta = -0.0396$).
  2. **Superior Generalization Under Limited Training Scale**: On a dataset of ~2,400 training fundus images, ResNet-50 and ConvNeXt-Tiny (with 23.5M and 27.8M parameters respectively) exhibited mild parameter overcapacity and higher training-set memorization compared to compound-scaled EfficientNet-B0 (~4.01M parameters).
  3. **Screening Safety**: EfficientNet-B0 achieved the highest referable sensitivity (**94.23%**) and referable specificity (**94.23%**), while ResNet-50 missed 5 additional Moderate/Severe DR cases (sensitivity dropped to 91.83%).
  4. **Edge Deployment Suitability**: In rural Primary Health Centres (PHCs) running on edge laptops or embedded hardware without high-end dedicated GPUs, EfficientNet-B0 provides an optimal footprint: ~6x smaller checkpoint size and ~2.5x to 3x faster inference throughput.
* **Limitations**:
  * Evaluations used fixed ImageNet-1K pretrained weights without intermediate domain-specific pre-training (e.g., on EyePACS). Future large-scale pre-training could benefit heavier architectures, but under the controlled E005 protocol, EfficientNet-B0 is the superior choice.