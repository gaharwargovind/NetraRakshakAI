# Robustness Validation & Quality Stress Testing (E011)

**Target Model:** `models/checkpoints/E007_best_model.pt`  
**Evaluation Scope:** Synthetic tele-screening perturbation stress tests across $N=601$ validation images.  
**Temperature Scaling:** Frozen at $T = 0.7785$.  

---

## 1. Perturbation Hierarchy & Sensitivity Thresholds
Controlled perturbations simulate real-world acquisition hazards in rural Indian screening camps (defocus blur, sensor thermal noise, illumination variations, and aggressive tele-ophthalmology JPEG compression).

### First Clinical Failure Points (Sensitivity < 90%):
- **Gaussian Blur:** Defocus beyond Level 2 ($k \ge 9, \sigma \ge 2.0$) drops sensitivity below 90%. Microaneurysms and faint dot hemorrhages are blurred into retinal background parenchyma.
- **Gaussian Noise:** Sensor noise beyond Level 2 ($\sigma \ge 25$) corrupts fine lesion features.
- **Brightness & Contrast:** Underexposure ($0.4\times$) severely impairs sensitivity by obscuring vascular landmarks.
- **JPEG Compression:** Extreme compression ($Q \le 20$) introduces macroblock boundary artifacts that degrade sensitivity.

## 2. Confidence Calibration Under Distribution Shift
Frozen temperature scaling ($T=0.7785$) consistently reduces ECE relative to raw probabilities across mild and moderate degradation. Under extreme failure conditions, ECE rises as model confidence disconnects from empirical accuracy, providing an effective out-of-distribution detection signal.
