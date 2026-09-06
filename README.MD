<!-- FILE: README.md -->
# Explainable AI for Diabetic Retinopathy Screening in Rural India
**Smart India Hackathon (SIH) — Problem Statement SIH26038**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Phase: Phase 1 Complete](https://img.shields.io/badge/Status-Phase%201%20Foundation%20Complete-brightgreen.svg)]()

---

## 1. Problem Statement Summary
India has a very large population living with diabetes, creating a substantial diabetic retinopathy screening burden[cite: 1]. Rural and resource-constrained healthcare settings have limited access to ophthalmologists and specialist infrastructure[cite: 1]. 

This project provides an AI-assisted retinal screening solution designed for Primary Health Centres (PHCs) and Vision Centres across rural India. The system evaluates fundus image quality, triggers immediate recapture for ungradable captures, performs 5-class ICDR severity grading, identifies referable disease ($\text{ICDR} \ge 2$), computes calibrated confidence and predictive uncertainty, generates visual evidence (Grad-CAM), routes ambiguous cases to tele-ophthalmologists, compiles bilingual referral documentation, and models population-level scalability ($\ge 100,000$ patients/year) using MATLAB and Simulink[cite: 1].

> **Clinical Safety & Legal Disclaimer**: This repository is a research and hackathon decision-support prototype. It does **not** provide definitive medical diagnosis, prescribe therapeutic treatments, or replace formal indirect ophthalmoscopy performed by a certified ophthalmologist.

---

## 2. Current Project Phase
* **Current Phase**: **Phase 1 — Repository & Tooling Foundation**
* **Completed**: Phase 0 (Problem Definition, Architecture, Clinical Workflow, Governance Audit), Phase 1 (Tooling, Configuration, Interfaces, Test Infrastructure).
* **Next Phase**: **Phase 2 — Dataset Acquisition & Multi-Source Audit**.
* **Metrics Notice**: All reported clinical metrics (e.g., Sensitivity $\ge 90\%$, Specificity $\ge 85\%$, $\text{ECE} \le 0.05$) are **Target Acceptance Criteria**; no models have been trained and no performance has been achieved yet[cite: 1].

---

## 3. System Architecture Overview

[Fundus Image (OD/OS)]
│
▼
┌──────────────────┐    UNGRADABLE    ┌───────────────────────────────────┐
│ Module A: IQA    │ ───────────────> │ Recapture Loop (Configurable Max) │
└─────────┬────────┘                  └─────────────────┬─────────────────┘
│ GRADABLE                                    ▼
▼                              Tele-Ophthalmology Priority Queue
┌──────────────────┐
│ Module B: Preproc│ (Crop, Resize 512x512, Normalization, CLAHE)
└─────────┬────────┘
▼
┌──────────────────┐
│ Module C: DR Net │ (ICDR 0–4 Ordinal Multi-Class Backbone)
└─────────┬────────┘
▼
┌──────────────────┐
│ Module D: Triage │ (Referable DR: Grade ≥ 2)
└─────────┬────────┘
│
┌────┴───────────────────────────┐
▼                                ▼
┌──────────────────┐       ┌──────────────────────┐
│ Module E: Saliency│      │ Module F: Calibration│
│ (Grad-CAM Map)   │       │ & Predictive Entropy │
└────┬─────────────┘       └──────────┬───────────┘
│                                │
└───────────────┬────────────────┘
▼
┌───────────────────────────────────────┐
│ Module G: Tele-Triage Router          │
│ (Routine PHC vs. Priority Specialist) │
└────────────────────┬──────────────────┘
▼
┌───────────────────────────────────────┐
│ Module H: Bilingual Referral Slip     │
└───────────────────────────────────────┘

---

## 4. Repository Structure & Navigation
* `configs/`: YAML experiment definitions (base, dataset-specific, E001–E010).
* `data/`: Raw, processed, and external dataset staging directories (git-ignored).
* `notebooks/`: Structured Jupyter exploratory templates.
* `src/`: Core Python library (`data`, `preprocessing`, `models`, `training`, `evaluation`, `explainability`, `robustness`, `inference`).
* `models/`: Model checkpoints and temperature-scaling calibration parameters.
* `experiments/`: Experiment ladder documentation and run configurations (E001–E010).
* `matlab/`: MATLAB edge preprocessing and ONNX inference verification scripts.
* `simulink/`: SimEvents discrete-event regional tele-screening simulation models.
* `tests/`: Automated unit and integration test suite.

---

## 5. Environment Setup & Tooling

### Prerequisites
* Python 3.11 or higher
* Conda or Python `venv`

### Conda Installation
```bash
conda env create -f environment.yml
conda activate dr-screening
pip install -e .

