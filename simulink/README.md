# NetraRakshakAI: Simulink Regional Systems Verification Architecture

## 1. System Engineering Role & Boundary
* **Primary Validated Screening Engine:** The clinical screening system is implemented and validated in the frozen **PyTorch E007 (EfficientNet-B0) + E013 deterministic IQA gate + E015 FastAPI orchestration pipeline**.
* **Simulink Role:** Macro-level systems engineering artifacts modeling workflow topology, signal-flow delays, and store-and-forward edge buffering under rural network outages.
* **Important Disclaimer:** Simulation assumptions are scenario parameters and must not be interpreted as measured clinical prevalence, clinical workflow observations, or deployment guarantees.

## 2. Model Designations & Queueing Scope
1. **`district_baseline_100k.slx` (System-Level Architectural Signal-Flow Model):**
   * Built via `build_baseline_model.m` using native Simulink blocks.
   * Visualizes macro workflow stages: Aggregate Arrivals, Acquisition, IQA Gate, Uplink Transfer, E007 Forward Pass (10.2 ms), Referable Branching, and Specialist Workload Accumulation.
   * **Important Queueing Distinction:** `district_baseline_100k.slx` is an architectural signal-flow model. It does **NOT** implement an individual-patient $M/G/5$ entity queue.
   * **Numerical Queueing Origin:** All numerical $M/G/5$ queue wait-time distributions, percentiles, and server utilization come from the event-driven simulation in `matlab/simulation/run_baseline_analysis.m`.
2. **`bandwidth_stress_test.slx` (Continuous-Rate Fluid Approximation Model):**
   * Built via `build_bandwidth_stress_model.m`.
   * The bandwidth model is a **continuous-rate fluid approximation, not a packet-level simulation**.
   * Implements fluid buffering: $\text{Buffer Rate} = \text{Influx Rate} - \text{Transmitted Rate}$, with active 4-hour blackout switching and a strict 64 GB upper saturation ceiling.

## 3. Programmatic Model Generation
To build the `.slx` files natively in MATLAB:
```matlab
cd simulink/scripts
build_baseline_model();          % Builds district_baseline_100k.slx
build_bandwidth_stress_model();   % Builds bandwidth_stress_test.slx