"""Production image quality gate screening interface and E013 Evaluator."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.data.quality import FundusIQAExtractor, IQAMetrics


class QualityGateDecision:
    """Logical decision contract for quality gate screening."""

    def __init__(self, max_recapture_attempts: int = 3) -> None:
        self.max_attempts = max_recapture_attempts

    def check(self, image: np.ndarray, attempt_count: int = 1) -> Tuple[bool, str, str]:
        raise NotImplementedError("Quality Gate decision engine is scheduled for Phase 13.")


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class MetricThreshold:
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    threshold_class: str = "experimental"
    clinical_note: str = ""


@dataclass
class QualityGateConfig:
    # Calibrated to empirical APTOS validation percentiles (P01/P99)
    fov_coverage: MetricThreshold = MetricThreshold(
        min_val=0.30, max_val=1.00, threshold_class="engineering",
        clinical_note="Catastrophic misframing or complete camera occlusion."
    )
    laplacian_variance: MetricThreshold = MetricThreshold(
        min_val=4.0, threshold_class="experimental",
        clinical_note="Distinguishes severe defocus/motion blur (Clean P01=4.14)."
    )
    edge_density: MetricThreshold = MetricThreshold(
        min_val=0.0005, threshold_class="experimental",
        clinical_note="Ensures visible retinal vessel network details (Clean P01=0.0008)."
    )
    mean_intensity: MetricThreshold = MetricThreshold(
        min_val=30.0, max_val=160.0, threshold_class="experimental",
        clinical_note="Bounds overall underexposure and overexposure (Clean P01=39.1, P99=130.7)."
    )
    dark_fraction: MetricThreshold = MetricThreshold(
        max_val=0.15, threshold_class="experimental",
        clinical_note="Rejects massive peripheral or macular signal dropout (Clean P99=0.01)."
    )
    bright_fraction: MetricThreshold = MetricThreshold(
        max_val=0.05, threshold_class="experimental",
        clinical_note="Rejects severe sensor flash washouts (Clean P99=0.00)."
    )
    percentile_spread_90: MetricThreshold = MetricThreshold(
        min_val=20.0, threshold_class="experimental",
        clinical_note="Enforces adequate tonal dynamic range across retina (Clean P01=23.0)."
    )
    noise_mad: MetricThreshold = MetricThreshold(
        max_val=4.0, threshold_class="experimental",
        clinical_note="Catches high-gain ISO noise artifacts (Clean P99=1.48)."
    )


@dataclass
class GateEvaluationResult:
    status: GateStatus
    rejection_reasons: List[str] = field(default_factory=list)
    metrics: Optional[IQAMetrics] = None


class FundusQualityGate:
    """Evaluates raw fundus images against engineering and experimental criteria."""

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()
        self.extractor = FundusIQAExtractor()

    def evaluate(self, img_bgr: np.ndarray) -> GateEvaluationResult:
        metrics = self.extractor.extract_metrics(img_bgr)
        reasons = []

        if self.config.fov_coverage.min_val and metrics.fov_coverage < self.config.fov_coverage.min_val:
            reasons.append(f"Low FOV coverage ({metrics.fov_coverage:.3f} < {self.config.fov_coverage.min_val})")
        if self.config.fov_coverage.max_val and metrics.fov_coverage > self.config.fov_coverage.max_val:
            reasons.append(f"Excessive FOV fill ({metrics.fov_coverage:.3f} > {self.config.fov_coverage.max_val})")

        if self.config.laplacian_variance.min_val and metrics.laplacian_variance < self.config.laplacian_variance.min_val:
            reasons.append(f"Severe defocus/blur (LapVar {metrics.laplacian_variance:.1f} < {self.config.laplacian_variance.min_val})")
        if self.config.edge_density.min_val and metrics.edge_density < self.config.edge_density.min_val:
            reasons.append(f"Insufficient edge density ({metrics.edge_density:.4f} < {self.config.edge_density.min_val})")

        if self.config.mean_intensity.min_val and metrics.mean_intensity < self.config.mean_intensity.min_val:
            reasons.append(f"Underexposure (Mean {metrics.mean_intensity:.1f} < {self.config.mean_intensity.min_val})")
        if self.config.mean_intensity.max_val and metrics.mean_intensity > self.config.mean_intensity.max_val:
            reasons.append(f"Overexposure (Mean {metrics.mean_intensity:.1f} > {self.config.mean_intensity.max_val})")
        if self.config.dark_fraction.max_val and metrics.dark_fraction > self.config.dark_fraction.max_val:
            reasons.append(f"High dark fraction ({metrics.dark_fraction:.3f} > {self.config.dark_fraction.max_val})")
        if self.config.bright_fraction.max_val and metrics.bright_fraction > self.config.bright_fraction.max_val:
            reasons.append(f"High saturation fraction ({metrics.bright_fraction:.3f} > {self.config.bright_fraction.max_val})")

        if self.config.percentile_spread_90.min_val and metrics.percentile_spread_90 < self.config.percentile_spread_90.min_val:
            reasons.append(f"Low contrast spread ({metrics.percentile_spread_90:.1f} < {self.config.percentile_spread_90.min_val})")

        if self.config.noise_mad.max_val and metrics.noise_mad > self.config.noise_mad.max_val:
            reasons.append(f"Excessive sensor noise (MAD {metrics.noise_mad:.2f} > {self.config.noise_mad.max_val})")

        status = GateStatus.PASS if len(reasons) == 0 else GateStatus.FAIL
        return GateEvaluationResult(status=status, rejection_reasons=reasons, metrics=metrics)
