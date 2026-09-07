# NetraRakshakAI Screening API Specification (Phase 16A)

## 1. Overview
Exposes the frozen E015 diabetic retinopathy screening pipeline over HTTP via FastAPI.

## 2. Local Startup
```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

Endpoints:
- GET /api/v1/health
- POST /api/v1/screen
- GET /api/v1/artifacts/{filename}
