"""Pydantic schemas for the NetraRakshakAI Phase 16A FastAPI interface."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    api_version: str
    model_checkpoint_available: bool
    model_checkpoint_sha256: str
    checkpoint_verified: bool
    device: str
    timestamp: str


class QualityResultSchema(BaseModel):
    status: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    failed_checks: List[str] = Field(default_factory=list)
    message: str


class ClassificationResultSchema(BaseModel):
    predicted_grade: int
    predicted_grade_name: str
    class_probabilities: List[float]
    referable: bool
    referable_probability: float
    confidence: float


class CalibrationSchema(BaseModel):
    temperature: float = 0.7785
    calibrated: bool
    method: Optional[str] = None


class ExplanationSchema(BaseModel):
    gradcam_available: bool
    target_layer: str = "model.model.features[-1]"
    status: str
    overlay_url: Optional[str] = None
    overlay_base64: Optional[str] = None


class RecommendationSchema(BaseModel):
    action: str
    reason: str


class TimingSchema(BaseModel):
    total_ms: float


class MetadataSchema(BaseModel):
    model_version: str = "E007"
    model_checkpoint_sha256: str
    timestamp: str
    pipeline_version: str = "1.0.0-E015"
    scientific_mandate: str


class ScreeningResponse(BaseModel):
    status: str
    image_id: str
    quality: QualityResultSchema
    classification: Optional[ClassificationResultSchema] = None
    calibration: Optional[CalibrationSchema] = None
    explanation: Optional[ExplanationSchema] = None
    recommendation: RecommendationSchema
    timing: TimingSchema
    metadata: MetadataSchema
