# MATLAB Integration & Verification Engine
**Domain**: Image Processing Toolbox & Deep Learning Toolbox  
**Status**: Foundation Scaffolding (Phase 1) — Implementation in Phase 17

### Purpose & Scope
This directory houses the MATLAB runtime integration verifying edge preprocessing equivalence, ONNX model ingestion, and automated clinical report generation:
1. `preprocessing/`: MATLAB-native implementation of circular crop, green-channel filtering, and CLAHE normalization.
2. `inference/`: `importNetworkFromONNX` model ingestion verifying numerical execution parity ($\|y_{\text{MATLAB}} - y_{\text{PyTorch}}\|_\infty \le 10^{-4}$) against the frozen PyTorch candidate.