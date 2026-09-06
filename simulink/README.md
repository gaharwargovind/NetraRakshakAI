# Simulink Regional Network Queuing Simulation
**Domain**: SimEvents & Stateflow Discrete-Event Modeling  
**Status**: Foundation Scaffolding (Phase 1) — Implementation in Phase 18

### Purpose & Scope
This directory houses the discrete-event model simulating rural tele-screening operations:
* **Target Scale**: $\ge 100,000$ patient encounters over a 1-year operational horizon[cite: 1].
* **Network Topology**: 50 rural PHC hubs feeding into 1 District Hospital Tele-Ophthalmology Review Centre.
* **Key Modeled Dynamics**: Poisson patient arrivals, edge IQA recapture loops, variable rural uplink telemetry (50 kbps–2 Mbps), local edge compute latencies, and specialist adjudication queues.