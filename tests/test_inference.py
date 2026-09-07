"""Unit tests for Phase E015 end-to-end inference pipeline and failure modes."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import torch

from src.inference.predictor import ScreeningPredictor, EXPECTED_E007_SHA
from src.inference.report_generator import compile_screening_report, compile_text_summary


@pytest.fixture(scope="module")
def predictor():
    return ScreeningPredictor(device="cpu")


@pytest.fixture
def clean_synthetic_fundus():
    """Generates an in-bounds synthetic fundus image (512x512)."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    radius = 210

    y, x = np.ogrid[:512, :512]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist <= radius

    base_val = 150 - (dist / radius * 80)
    img[mask, 0] = (base_val[mask] * 0.2).astype(np.uint8)
    img[mask, 1] = (base_val[mask] * 0.6).astype(np.uint8)
    img[mask, 2] = (base_val[mask] * 1.0).astype(np.uint8)
    return img


def test_checkpoint_hash_invariant(predictor):
    assert predictor.checkpoint_sha == EXPECTED_E007_SHA


def test_schema_conformance_pass(predictor, clean_synthetic_fundus):
    result = predictor.predict(clean_synthetic_fundus, image_id="test_pass_001")

    assert "image_id" in result
    assert "quality" in result
    assert "classification" in result
    assert "calibration" in result
    assert "explanation" in result
    assert "recommendation" in result
    assert "metadata" in result

    assert result["quality"]["status"] == "PASS"
    assert result["classification"] is not None
    assert 0 <= result["classification"]["predicted_grade"] <= 4
    assert len(result["classification"]["class_probabilities"]) == 5
    assert isinstance(result["classification"]["referable"], bool)
    assert 0.0 <= result["classification"]["confidence"] <= 1.0
    assert result["calibration"]["temperature"] == 0.7785
    assert result["metadata"]["model_checkpoint_sha256"] == EXPECTED_E007_SHA


def test_iqa_failure_skips_inference(predictor):
    # Solid black image triggers FOV and underexposure failure
    black_img = np.zeros((512, 512, 3), dtype=np.uint8)
    result = predictor.predict(black_img, image_id="test_fail_001")

    assert result["quality"]["status"] == "FAIL"
    assert result["classification"] is None
    assert result["recommendation"]["action"] == "RECAPTURE_OR_HUMAN_REVIEW"
    assert len(result["quality"]["failed_checks"]) > 0


def test_referable_logic_consistency(predictor, clean_synthetic_fundus):
    result = predictor.predict(clean_synthetic_fundus)
    cls = result["classification"]
    if cls["predicted_grade"] >= 2:
        assert cls["referable"] is True
        assert result["recommendation"]["action"] == "SPECIALIST_REFERRAL"
    else:
        assert cls["referable"] is False


def test_nan_inf_input_handling(predictor):
    corrupt = np.full((512, 512, 3), np.nan)
    res = predictor.predict(corrupt)
    assert res["quality"]["status"] == "FAIL"
    assert res["classification"] is None
    assert "NaN or Inf" in res["quality"]["failed_checks"][0]


def test_grayscale_input_conversion(predictor):
    gray = np.full((512, 512), 128, dtype=np.uint8)
    res = predictor.predict(gray)
    assert "quality" in res


def test_invalid_dimensions_rejection(predictor):
    bad_dim = np.zeros((512, 512, 7), dtype=np.uint8)
    res = predictor.predict(bad_dim)
    assert res["quality"]["status"] == "FAIL"
    assert res["classification"] is None


def test_none_input_rejection(predictor):
    res = predictor.predict(None)
    assert res["quality"]["status"] == "FAIL"
    assert res["classification"] is None


def test_deterministic_reproducibility(predictor, clean_synthetic_fundus):
    res1 = predictor.predict(clean_synthetic_fundus, image_id="repeat_01", generate_saliency=False)
    res2 = predictor.predict(clean_synthetic_fundus, image_id="repeat_01", generate_saliency=False)

    assert res1["classification"]["predicted_grade"] == res2["classification"]["predicted_grade"]
    assert res1["classification"]["class_probabilities"] == res2["classification"]["class_probabilities"]


def test_report_compilation_pdf_and_text(tmp_path, predictor, clean_synthetic_fundus):
    res = predictor.predict(clean_synthetic_fundus, image_id="rep_test_01")

    txt_path = tmp_path / "report.txt"
    out_txt = compile_screening_report(res, txt_path)
    assert out_txt.is_file()
    assert "MANDATORY SCIENTIFIC & REGULATORY NOTICE" in out_txt.read_text()

    pdf_path = tmp_path / "report.pdf"
    out_pdf = compile_screening_report(res, pdf_path)
    assert out_pdf.is_file()
    assert out_pdf.stat().st_size > 500
