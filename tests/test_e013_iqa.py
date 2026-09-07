"""Unit tests for Phase E013 deterministic IQA feature extraction and quality gate."""

import pytest
import numpy as np
import cv2
from src.data.quality import FundusIQAExtractor, IQAMetrics
from src.inference.quality_gate import FundusQualityGate, GateStatus, QualityGateConfig


@pytest.fixture
def synthetic_fundus():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    radius = 210

    y, x = np.ogrid[:512, :512]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius

    base_val = 160 - (dist_from_center / radius * 100)
    img[mask, 0] = (base_val[mask] * 0.2).astype(np.uint8)
    img[mask, 1] = (base_val[mask] * 0.6).astype(np.uint8)
    img[mask, 2] = (base_val[mask] * 1.0).astype(np.uint8)

    disc_mask = np.sqrt((x - 320)**2 + (y - 256)**2) <= 35
    img[disc_mask, 0] = 80
    img[disc_mask, 1] = 180
    img[disc_mask, 2] = 240

    cv2.line(img, (320, 256), (200, 180), (10, 20, 90), 4)
    cv2.line(img, (200, 180), (120, 160), (10, 20, 90), 2)
    cv2.line(img, (320, 256), (200, 320), (10, 20, 90), 4)
    cv2.line(img, (200, 320), (120, 340), (10, 20, 90), 2)
    return img


def test_retinal_mask_extraction(synthetic_fundus):
    extractor = FundusIQAExtractor()
    mask = extractor.extract_retinal_mask(synthetic_fundus)
    assert mask.shape == (512, 512)
    assert np.count_nonzero(mask) > 100000
    assert mask[0, 0] == 0
    assert mask[256, 256] == 255


def test_metrics_extraction_ranges(synthetic_fundus):
    extractor = FundusIQAExtractor()
    m = extractor.extract_metrics(synthetic_fundus)
    assert 0.40 <= m.fov_coverage <= 0.70
    assert m.laplacian_variance > 4.0
    assert m.edge_density > 0.0005
    assert 30.0 <= m.mean_intensity <= 160.0
    assert m.dark_fraction < 0.15
    assert m.bright_fraction < 0.05
    assert m.illumination_uniformity > 0.60
    assert m.percentile_spread_90 > 20.0
    assert m.noise_mad >= 0.0


def test_engineering_threshold_solid_black():
    black_img = np.zeros((512, 512, 3), dtype=np.uint8)
    gate = FundusQualityGate()
    res = gate.evaluate(black_img)
    assert res.status == GateStatus.FAIL
    assert any("FOV" in r or "Underexposure" in r for r in res.rejection_reasons)


def test_blur_sensitivity(synthetic_fundus):
    gate = FundusQualityGate()
    res_clean = gate.evaluate(synthetic_fundus)
    assert res_clean.status == GateStatus.PASS

    # Extreme defocus blur reduces Laplacian variance below 4.0
    blurred = cv2.GaussianBlur(synthetic_fundus, (45, 45), 18.0)
    res_blur = gate.evaluate(blurred)
    assert res_blur.status == GateStatus.FAIL
    assert any("defocus" in r or "blur" in r for r in res_blur.rejection_reasons)


def test_severe_noise_sensitivity(synthetic_fundus):
    gate = FundusQualityGate()
    np.random.seed(42)
    noise = np.random.normal(0, 30, synthetic_fundus.shape).astype(np.float32)
    noisy = np.clip(synthetic_fundus.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    res_noisy = gate.evaluate(noisy)
    assert res_noisy.status == GateStatus.FAIL
    assert any("sensor noise" in r for r in res_noisy.rejection_reasons)


def test_severe_underexposure(synthetic_fundus):
    gate = FundusQualityGate()
    dark = (synthetic_fundus.astype(np.float32) * 0.15).astype(np.uint8)
    res_dark = gate.evaluate(dark)
    assert res_dark.status == GateStatus.FAIL
    assert any("Underexposure" in r or "dark fraction" in r for r in res_dark.rejection_reasons)


def test_severe_saturation(synthetic_fundus):
    gate = FundusQualityGate()
    extractor = FundusIQAExtractor()
    mask = extractor.extract_retinal_mask(synthetic_fundus) > 0
    saturated = synthetic_fundus.copy()
    saturated[mask] = np.clip(saturated[mask].astype(np.float32) * 2.5 + 80, 0, 255).astype(np.uint8)
    res_sat = gate.evaluate(saturated)
    assert res_sat.status == GateStatus.FAIL
    assert any("Overexposure" in r or "saturation" in r for r in res_sat.rejection_reasons)


def test_e007_invariance_declaration():
    config = QualityGateConfig()
    assert not hasattr(config, "referable_prob_threshold")
    assert not hasattr(config, "p_referable_cutoff")
