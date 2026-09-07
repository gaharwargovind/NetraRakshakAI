# Phase E012 Causal Claim Audit

## Audit Overview
This document evaluates causal and explanatory claims made during Phase E012 (External Validation on IDRiD).

## Claim Classifications

### Claim 1: E007 External Sensitivity Drop
- **Statement:** "E007 exhibits significant sensitivity degradation (79.61%) on IDRiD compared to internal validation, failing the safety gate."
- **Classification:** SUPPORTED
- **Basis:** Directly measured on IDRiD cohort (N=455).

### Claim 2: Specificity Invariance
- **Statement:** "High referral specificity is maintained across domains (98.68% on IDRiD)."
- **Classification:** SUPPORTED
- **Basis:** Directly measured on IDRiD cohort.

### Claim 3: False Negatives Concentrated in Grade 2
- **Statement:** "Sensitivity drop is driven primarily by false-negative errors in Grade 2 (Moderate NPDR)."
- **Classification:** SUPPORTED
- **Basis:** Contingency matrix confirms 76 of 93 false negatives occurred in Grade 2.

### Claim 4: Annotation Threshold Divergence
- **Statement:** "Differences in clinical annotation criteria between Indian and European grading practices caused the Grade 2 drop."
- **Classification:** PARTIALLY SUPPORTED
- **Basis:** Plausible from literature, but no dual-read experiment was conducted.

### Claim 5: Camera Hardware Shift
- **Statement:** "Domain shift is caused by optical and sensor differences between Kowa VX-10alpha (IDRiD) and Topcon TRC-NW (APTOS)."
- **Classification:** PARTIALLY SUPPORTED
- **Basis:** Hardware is documented, but optical transfer functions were not independently isolated.

### Claim 6: Lesion Size Filtering
- **Statement:** "E007 downsampling layers filtered out microaneurysms smaller than 3 pixels on IDRiD."
- **Classification:** HYPOTHESIS
- **Basis:** Plausible mechanism, but pixel-level feature activations were not tracked.

### Claim 7: Choroidal Pigmentation Bias
- **Statement:** "Choroidal background contrast differences in Indian cohorts depressed early lesion saliency."
- **Classification:** HYPOTHESIS
- **Basis:** Plausible retinal physiology, but no pigmentation sub-stratification was evaluated.

### Claim 8: Threshold Invariant Fixability
- **Statement:** "Tuning the referable decision threshold P(referable) can restore IDRiD clinical safety."
- **Classification:** UNSUPPORTED
- **Basis:** Retrospective tuning on test data violates governance and collapses specificity below 85%.

### Claim 9: Synthetic Degradation Equivalence
- **Statement:** "Images failing E011 synthetic perturbation checks reflect ungradable images in IDRiD."
- **Classification:** UNSUPPORTED
- **Basis:** E011 tested mathematical global perturbations, whereas clinical ungradability involves focal biological artifacts.
