# Phase 4 - Preprocessing Audit

## Status

Phase: 4 - Preprocessing
Dataset: APTOS 2019 Blindness Detection
Status: IN PROGRESS
Date: 2026-09-05

## 1. Raw Image Audit

Representative APTOS images were visually inspected across all five DR
severity grades.

Observed characteristics:

- Retinal fields are generally circular or elliptical.
- Dark/black background regions are common.
- Image framing varies substantially.
- Source image resolutions are heterogeneous.
- Illumination varies between images.
- Retinal vessel and lesion visibility varies between images.
- Acquisition artifacts are present in some images.

## 2. Baseline Preprocessing

The baseline preprocessing pipeline is:

raw RGB image
-> retinal-field crop
-> resize to 512x512
-> float32 conversion
-> pixel normalization to [0,1]

The baseline does not apply CLAHE or illumination normalization.

## 3. Retinal Cropping

Retinal-field cropping was visually evaluated on representative images.

The crop substantially reduces surrounding dark background while retaining the
main retinal field.

Decision:

KEEP AS BASELINE

Further automated evaluation across the complete dataset is still required.

## 4. CLAHE

CLAHE was visually evaluated using representative images from DR grades 0-4.

CLAHE increases local contrast and makes retinal vessels and some lesions more
prominent.

However, it also increases visible texture and may amplify image noise or
artifacts.

Decision:

EXPERIMENTAL ONLY

CLAHE will not be included in the baseline until validation experiments
determine whether it improves model performance.

## 5. Illumination Normalization

The initial implementation used local color subtraction based on a Gaussian
illumination estimate.

Visual evaluation showed that the resulting images had a strong
edge/high-pass appearance rather than preserving a natural fundus
photograph appearance.

Decision:

CURRENT IMPLEMENTATION REJECTED

It must not be used by the baseline model.

A different illumination-correction formulation may be investigated as a
future experiment.

## 6. Current Baseline

The current baseline is:

crop -> resize -> normalize

Optional transformations remain disabled:

- illumination normalization
- CLAHE
- denoising

## 7. Experimental Principle

Visual improvement alone does not establish clinical or model usefulness.

Preprocessing transformations must ultimately be evaluated using the
validation set and compared against the baseline under the project's
experiment-control rules.

## 8. Remaining Phase 4 Work

- [x] Raw image visual audit
- [x] Retinal crop implementation
- [x] Baseline pipeline implementation
- [x] CLAHE implementation
- [x] Initial CLAHE visual evaluation
- [x] Initial illumination visual evaluation
- [x] Reject unsuitable illumination implementation
- [ ] Unit tests for preprocessing
- [ ] Crop robustness audit
- [ ] Batch preprocessing benchmark
- [ ] Finalize preprocessing experiment configuration
- [ ] Compare preprocessing variants on validation data

## Conclusion

The current evidence supports using crop, resize, and normalization as the
initial preprocessing baseline.

CLAHE remains an experimental candidate.

The current subtraction-based illumination implementation is rejected from
the baseline because its visual output is not an acceptable natural fundus
representation.

PHASE 4 STATUS: IN PROGRESS
