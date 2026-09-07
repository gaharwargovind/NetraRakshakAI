# Image Quality Assessment (IQA) Validation (Phase E013)

## 1. Upstream Engineering Role
IQA functions as a pre-inference deterministic safety triage. Images failing the gate are flagged for recapture or expert human review rather than unmonitored automated inference.

> **Operational Scope:** IQA provides preliminary engineering evidence for identifying potentially compromised images; clinical validation of gradability remains necessary.

---

## 2. Terminology & Epistemic Distinctions
To preserve scientific rigor, three distinct operational states must not be conflated:

1. **Clean-Validation Rejection:** The proportion of in-distribution, uncorrupted validation images rejected by experimental IQA boundaries (10.65%, 64/601). This represents an operational engineering exclusion rate, not a clinical false alarm.
2. **Synthetic Degradation Rejection:** The proportion of mathematically perturbed images rejected across controlled parameter sweeps.
3. **Clinical Ungradability:** The professional ophthalmological determination that an eye cannot be clinically assessed due to media opacities, severe optical artifacts, or inadequate field definition. Clean validation images in APTOS lack adjudicated clinical gradability labels; therefore, rejected clean images must never be labeled clinically ungradable.

---

## 3. Threshold Taxonomy
- **Tier 1 (Engineering):** Hard physical and geometric constraints (e.g., solid sensor loss, total occlusion).
- **Tier 2 (Experimental):** Statistical boundaries derived from clean APTOS validation distributions (P01/P99 tails).
- **Tier 3 (Clinical):** None established. No threshold in this experiment is clinically validated.

---

## 4. Known Limitations: JPEG Compression Non-Monotonicity
Empirical stress-testing under JPEG compression revealed non-monotonic rejection behavior:
- **Q=80:** 12.48% rejection
- **Q=60:** 12.48% rejection
- **Q=40:** 29.95% rejection
- **Q=20:** 12.31% rejection
- **Q=10:** 0.83% rejection

### Mechanism of Failure:
Discrete Cosine Transform (DCT) quantization introduces sharp $8 \times 8$ pixel grid discontinuities (blocking artifacts). The Canny edge detector perceives these artificial grid boundaries as structural edges, artificially elevating the measured edge density metric ($D_{\text{edge}}$) on severely compressed inputs. Consequently, **no JPEG threshold may be derived from Canny edge density**, and a dedicated blocking/SSIM metric will be required.
