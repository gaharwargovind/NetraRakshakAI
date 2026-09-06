# Model Governance Card (Template)
**Target Model**: EfficientNet-B0 Multi-Class DR Screening Classifier  
**Status**: Pre-Training Template (Phase 1)

### Intended Domain
* **Clinical Task**: Decision-support screening for Diabetic Retinopathy from color fundus photographs.
* **Intended Users**: Trained healthcare workers and tele-ophthalmologists in rural primary care clinics.
* **Out-of-Scope Use**: Autonomous medical diagnosis, prescribing medications, direct surgical triage.

### Performance Target Gates (Not Yet Evaluated)
* Referable DR Sensitivity: $\ge 90.0\%$[cite: 1]
* Referable DR Specificity: $\ge 85.0\%$[cite: 1]
* Quadratic Weighted Kappa: $\ge 0.80$
* Expected Calibration Error (ECE): $\le 0.05$[cite: 1]