# Experiment E013: Image Quality Assessment (IQA) Benchmark

## Executive Summary
- **Status:** COMPLETE (PROVISIONALLY ACCEPTED)
- **Upstream Scope:** Deterministic pre-inference quality triage.
- **Clean-Validation Rejection Rate:** `10.65%` (64 / 601 clean validation images)
- **Downstream E007 Sensitivity (IQA-PASS):** `95.33%` (Unfiltered: `94.44%`)
- **Downstream E007 QWK (IQA-PASS):** `0.9239` (Unfiltered: `0.9156`)

## Core Safety Declaration
IQA provides preliminary engineering evidence for identifying potentially compromised images; clinical validation of gradability remains necessary.

## Terminology Note
Clean APTOS validation images lack adjudicated clinical gradability labels. Images rejected by the gate represent **clean-validation rejections**, not clinical ungradability.

## JPEG Blocking Limitation
At severe compression ($Q \le 20$), DCT block boundaries inflate Canny edge detection, reducing rejection rates to $0.83\%$. No JPEG operational threshold is derived from edge density.
