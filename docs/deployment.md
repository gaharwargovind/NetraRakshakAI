# Rural Deployment Architecture
## Primary Health Centre (PHC) & Vision Centre Operating Guidelines

### 1. Hardware Specification for Edge Deployment
* **Processor**: Minimum Quad-Core x86_64 or ARM64 (e.g., Intel Core i3 10th Gen+, Raspberry Pi 5 8GB, or embedded medical PC).
* **RAM**: 8 GB minimum (16 GB recommended).
* **Storage**: 256 GB SSD (encrypted with AES-256 for local image caching).
* **Display**: Minimum $1920 \times 1080$ resolution with sRGB color profile support.

### 2. Operational Continuity in Low-Resource Settings
* **Offline-First Execution**: The complete screening pipeline (IQA, Preprocessing, Classification, Grad-CAM, and Local Report Compilation) executes locally without cloud dependencies.
* **Store-and-Forward Telemetry**: When tele-consultation is required, cases are queued in a local encrypted SQLite database and synchronized as upstream bandwidth permits.
* **Power Resilience**: System must operate alongside a standard uninterrupted power supply (UPS) supporting $\ge 2$ hours of offline battery operation.