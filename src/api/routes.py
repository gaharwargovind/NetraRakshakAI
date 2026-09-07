"""FastAPI route handlers wrapping the E015 screening pipeline."""

import base64
from datetime import datetime, timezone
import io
from pathlib import Path
import time
from typing import Dict, Optional
import cv2
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.api.schemas import (
    CalibrationSchema,
    ClassificationResultSchema,
    ExplanationSchema,
    HealthResponse,
    MetadataSchema,
    QualityResultSchema,
    RecommendationSchema,
    ScreeningResponse,
    TimingSchema,
)
from src.inference.predictor import EXPECTED_E007_SHA

router = APIRouter(prefix="/api/v1")

ICDR_GRADE_NAMES: Dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request, response: Response) -> HealthResponse:
    """Returns runtime readiness and verifies E007 checkpoint integrity from loaded predictor state."""
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is not None:
        ckpt_path = getattr(predictor, "checkpoint_path", Path("models/checkpoints/E007_best_model.pt"))
        ckpt_exists = Path(ckpt_path).is_file()
        ckpt_sha = getattr(predictor, "checkpoint_sha", "unavailable")
    else:
        ckpt_exists = Path("models/checkpoints/E007_best_model.pt").is_file()
        ckpt_sha = "unavailable"

    checkpoint_verified = bool(
        predictor is not None
        and ckpt_exists
        and ckpt_sha == EXPECTED_E007_SHA
    )
    is_ready = bool(predictor is not None and checkpoint_verified)

    if is_ready:
        status_str = "ok"
    else:
        status_str = "not_ready"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    device = str(predictor.device) if predictor else "unloaded"

    return HealthResponse(
        status=status_str,
        api_version="1.0.0-phase16a",
        model_checkpoint_available=ckpt_exists,
        model_checkpoint_sha256=str(ckpt_sha),
        checkpoint_verified=checkpoint_verified,
        device=device,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/artifacts/{filename}")
async def get_artifact(filename: str):
    """Safely serves saved Grad-CAM overlay PNGs."""
    safe_name = Path(filename).name
    artifact_path = Path("artifacts/api_outputs") / safe_name
    if not artifact_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {safe_name} not found.",
        )
    return FileResponse(artifact_path, media_type="image/png")


@router.post("/screen", response_model=ScreeningResponse)
async def screen_fundus(
    request: Request,
    file: UploadFile = File(...),
) -> ScreeningResponse:
    """Invokes the existing E015 inference pipeline against an uploaded fundus image."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Screening engine is not initialized.",
        )

    content_type = file.content_type
    normalized_ct = (content_type or "").strip().lower()
    suffix = Path(file.filename or "").suffix.lower()

    if normalized_ct:
        if normalized_ct not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: '{content_type}'. Must be one of JPEG, PNG, WebP, BMP, or TIFF.",
            )
    else:
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: '{content_type}'. Must be one of JPEG, PNG, WebP, BMP, or TIFF.",
            )

    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file buffer.",
        )

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    try:
        pil_image = Image.open(io.BytesIO(raw_bytes))
        pil_image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unreadable image payload. File cannot be decoded.",
        )

    try:
        rgb_image = pil_image.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to convert image to RGB color space.",
        )

    clean_stem = Path(file.filename or "upload").stem.replace(" ", "_")
    image_id = f"{clean_stem}_{int(time.time() * 1000)}"

    start_time = time.perf_counter()
    try:
        e015_result = predictor.predict(
            image=rgb_image,
            image_id=image_id,
            generate_saliency=True,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error encountered during model inference.",
        )
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    quality_raw = e015_result.get("quality", {})
    quality_status = quality_raw.get("status", "FAIL")
    api_status = "SUCCESS" if quality_status == "PASS" else "IQA_FAIL"

    quality_schema = QualityResultSchema(
        status=quality_status,
        metrics=quality_raw.get("metrics", {}),
        failed_checks=quality_raw.get("failed_checks", []),
        message=quality_raw.get("message", ""),
    )

    cls_raw = e015_result.get("classification")
    classification_schema: Optional[ClassificationResultSchema] = None
    if cls_raw is not None:
        grade = int(cls_raw["predicted_grade"])
        classification_schema = ClassificationResultSchema(
            predicted_grade=grade,
            predicted_grade_name=ICDR_GRADE_NAMES.get(grade, "Unknown"),
            class_probabilities=cls_raw["class_probabilities"],
            referable=cls_raw["referable"],
            referable_probability=cls_raw["referable_probability"],
            confidence=cls_raw["confidence"],
        )

    calib_raw = e015_result.get("calibration", {})
    calibration_schema = CalibrationSchema(
        temperature=calib_raw.get("temperature", 0.7785),
        calibrated=calib_raw.get("calibrated", False),
        method=calib_raw.get("method"),
    )

    expl_raw = e015_result.get("explanation", {})
    overlay_arr = expl_raw.get("overlay")

    overlay_url: Optional[str] = None
    overlay_b64: Optional[str] = None

    if overlay_arr is not None and isinstance(overlay_arr, np.ndarray):
        try:
            success, png_buffer = cv2.imencode(".png", cv2.cvtColor(overlay_arr, cv2.COLOR_RGB2BGR))
            if success:
                png_bytes = png_buffer.tobytes()
                artifact_dir = Path("artifacts/api_outputs")
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact_file = artifact_dir / f"{image_id}_overlay.png"
                artifact_file.write_bytes(png_bytes)

                overlay_url = f"/api/v1/artifacts/{image_id}_overlay.png"
                b64_encoded = base64.b64encode(png_bytes).decode("ascii")
                overlay_b64 = f"data:image/png;base64,{b64_encoded}"
        except Exception:
            overlay_url = None
            overlay_b64 = None

    explanation_schema = ExplanationSchema(
        gradcam_available=expl_raw.get("gradcam_available", False),
        target_layer=expl_raw.get("target_layer", "model.model.features[-1]"),
        status=expl_raw.get("status", "unavailable"),
        overlay_url=overlay_url,
        overlay_base64=overlay_b64,
    )

    rec_raw = e015_result.get("recommendation", {})
    recommendation_schema = RecommendationSchema(
        action=rec_raw.get("action", "HUMAN_REVIEW"),
        reason=rec_raw.get("reason", ""),
    )

    meta_raw = e015_result.get("metadata", {})
    metadata_schema = MetadataSchema(
        model_version=meta_raw.get("model_version", "E007"),
        model_checkpoint_sha256=meta_raw.get("model_checkpoint_sha256", predictor.checkpoint_sha),
        timestamp=meta_raw.get("timestamp", datetime.now(timezone.utc).isoformat()),
        pipeline_version=meta_raw.get("pipeline_version", "1.0.0-E015"),
        scientific_mandate=meta_raw.get("scientific_mandate", ""),
    )

    return ScreeningResponse(
        status=api_status,
        image_id=image_id,
        quality=quality_schema,
        classification=classification_schema,
        calibration=calibration_schema,
        explanation=explanation_schema,
        recommendation=recommendation_schema,
        timing=TimingSchema(total_ms=elapsed_ms),
        metadata=metadata_schema,
    )
