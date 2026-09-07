"""Integration tests for the Phase 16A FastAPI screening service."""

import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.api.app import app
from src.inference.predictor import EXPECTED_E007_SHA


class SimpleASGIGateway:
    """Synchronous ASGI test client wrapper requiring zero external HTTP client libraries."""
    def __init__(self, app):
        self.app = app

    def get(self, path: str, headers=None):
        return self._request("GET", path, headers=headers)

    def post(self, path: str, headers=None, files=None):
        return self._request("POST", path, headers=headers, files=files)

    def _request(self, method: str, path: str, headers=None, files=None):
        headers = headers or []
        body = b""
        content_type = ""

        if files:
            field_name, (filename, file_bytes, mime) = list(files.items())[0]
            boundary = "----WebKitFormBoundaryTestBoundary"
            content_type = f"multipart/form-data; boundary={boundary}"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"content-type", content_type.encode("utf-8"))] + [(k.encode(), v.encode()) for k, v in headers],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
            "asgi": {"version": "3.0"},
            "state": app.state.__dict__,
        }

        response_status = 200
        response_headers = []
        response_body = bytearray()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = message.get("headers", [])
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If running inside an existing loop context, run synchronously via run_until_complete in a new thread or inline
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                def run_sync():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.app(scope, receive, send))
                    finally:
                        new_loop.close()
                pool.submit(run_sync).result()
        else:
            asyncio.run(self.app(scope, receive, send))

        class MockResponse:
            def __init__(self, status_code, content, headers):
                self.status_code = status_code
                self.content = content
                self.headers = {k.decode(): v.decode() for k, v in headers}

            def json(self):
                import json
                return json.loads(self.content.decode("utf-8"))

        return MockResponse(response_status, bytes(response_body), response_headers)


@pytest.fixture(scope="module")
def client():
    # Ensure app lifespan runs or app.state has predictor initialized
    if not hasattr(app.state, "predictor") or app.state.predictor is None:
        try:
            from src.inference.predictor import ScreeningPredictor
            app.state.predictor = ScreeningPredictor()
        except Exception:
            app.state.predictor = None
    return SimpleASGIGateway(app)


@pytest.fixture
def synthetic_fundus_bytes():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    radius = 210

    y, x = np.ogrid[:512, :512]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist <= radius

    base_val = 150 - (dist / radius * 80)
    img[mask, 0] = (base_val[mask] * 0.2).astype(np.uint8)
    img[mask, 1] = (base_val[mask] * 0.6).astype(np.uint8)
    img[mask, 2] = (base_val[mask] * 1.0).astype(np.uint8)

    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def black_image_bytes():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def find_sample_e015_image_path() -> Path | None:
    candidate_paths = [
        Path("data/raw/aptos/train_images"),
        Path("data/raw/aptos/test_images"),
        Path("experiments/E015_inference"),
    ]
    for directory in candidate_paths:
        if directory.is_dir():
            for p in sorted(directory.glob("*.png")):
                if p.stat().st_size > 0:
                    return p
            for p in sorted(directory.glob("*.jpg")):
                if p.stat().st_size > 0:
                    return p
    return None


def test_health_endpoint_ready(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["api_version"] == "1.0.0-phase16a"
    assert data["model_checkpoint_available"] is True
    assert data["model_checkpoint_sha256"] == EXPECTED_E007_SHA
    assert data["checkpoint_verified"] is True
    assert "device" in data


def test_health_endpoint_unready_returns_503(client):
    original_predictor = app.state.predictor
    try:
        app.state.predictor = None
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert data["checkpoint_verified"] is False
    finally:
        app.state.predictor = original_predictor


def test_valid_png_upload_accepted(client, synthetic_fundus_bytes):
    img = Image.open(io.BytesIO(synthetic_fundus_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    files = {"file": ("sample_eye.png", png_bytes, "image/png")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert data["quality"]["status"] == "PASS"


def test_invalid_mime_with_png_extension(client, synthetic_fundus_bytes):
    files = {"file": ("sample_eye.png", synthetic_fundus_bytes, "text/plain")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 415
    detail = response.json()["detail"]
    assert detail == "Unsupported file type: 'text/plain'. Must be one of JPEG, PNG, WebP, BMP, or TIFF."


def test_missing_mime_with_valid_png_extension(client, synthetic_fundus_bytes):
    img = Image.open(io.BytesIO(synthetic_fundus_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    files = {"file": ("sample_eye.png", png_bytes, "")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("SUCCESS", "IQA_FAIL")


def test_invalid_extension_with_missing_mime(client):
    files = {"file": ("document.pdf", b"%PDF-1.4 fake content", "")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 415
    detail = response.json()["detail"]
    assert "Must be one of JPEG, PNG, WebP, BMP, or TIFF." in detail


def test_invalid_upload_corrupt_bytes(client):
    files = {"file": ("corrupt.jpg", b"NOT_A_JPEG_RANDOM_CORRUPT_BYTES", "image/jpeg")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 400
    assert "Invalid or unreadable image payload" in response.json()["detail"]


def test_iqa_failure_path(client, black_image_bytes):
    files = {"file": ("underexposed.png", black_image_bytes, "image/png")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "IQA_FAIL"
    assert data["quality"]["status"] == "FAIL"
    assert len(data["quality"]["failed_checks"]) > 0
    assert data["classification"] is None
    assert data["explanation"]["status"] == "skipped"
    assert data["recommendation"]["action"] == "RECAPTURE_OR_HUMAN_REVIEW"
    assert data["timing"]["total_ms"] > 0


def test_successful_screening_path(client, synthetic_fundus_bytes):
    files = {"file": ("sample_eye.jpg", synthetic_fundus_bytes, "image/jpeg")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert data["quality"]["status"] == "PASS"
    assert data["classification"] is not None

    cls = data["classification"]
    assert 0 <= cls["predicted_grade"] <= 4
    assert cls["predicted_grade_name"] in [
        "No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"
    ]
    assert len(cls["class_probabilities"]) == 5
    assert isinstance(cls["referable"], bool)
    assert 0.0 <= cls["confidence"] <= 1.0

    assert data["calibration"]["temperature"] == 0.7785
    assert data["calibration"]["calibrated"] is True

    expl = data["explanation"]
    assert expl["gradcam_available"] is True
    assert expl["target_layer"] == "model.model.features[-1]"
    assert expl["overlay_url"] is not None
    assert expl["overlay_base64"] is not None
    assert expl["overlay_base64"].startswith("data:image/png;base64,")

    assert data["recommendation"]["action"] in [
        "SPECIALIST_REFERRAL", "HUMAN_REVIEW", "ROUTINE_MONITORING"
    ]
    assert data["metadata"]["model_checkpoint_sha256"] == EXPECTED_E007_SHA
    assert data["timing"]["total_ms"] > 0


def test_real_e015_fundus_image_integration(client):
    real_sample_path = find_sample_e015_image_path()
    if real_sample_path is None:
        pytest.skip("No real existing E015/APTOS fundus image available in local cohort path.")

    with open(real_sample_path, "rb") as f:
        file_bytes = f.read()

    mime = "image/png" if real_sample_path.suffix.lower() == ".png" else "image/jpeg"
    files = {"file": (real_sample_path.name, file_bytes, mime)}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("SUCCESS", "IQA_FAIL")
    assert "quality" in data
    assert "timing" in data
    assert data["metadata"]["model_checkpoint_sha256"] == EXPECTED_E007_SHA


def test_e015_invariants_preserved(client):
    predictor = app.state.predictor
    assert predictor is not None
    assert predictor.checkpoint_sha == EXPECTED_E007_SHA
    assert predictor.temperature == 0.7785


def test_upload_exceeds_size_limit(client, monkeypatch):
    import src.api.routes as routes_module
    monkeypatch.setattr(routes_module, "MAX_IMAGE_SIZE_BYTES", 10)
    large_bytes = b"0" * 20
    files = {"file": ("huge_scan.jpg", large_bytes, "image/jpeg")}
    response = client.post("/api/v1/screen", files=files)
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]