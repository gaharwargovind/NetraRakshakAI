"""Unit tests for retinal image preprocessing."""

import cv2
import numpy as np
import pytest

from src.preprocessing.clahe import apply_clahe
from src.preprocessing.crop_retina import crop_retina_circle
from src.preprocessing.illumination import normalize_illumination
from src.preprocessing.pipeline import PreprocessingPipeline


def make_synthetic_retina(
    height: int = 600,
    width: int = 800,
) -> np.ndarray:
    """Create a synthetic RGB image with a bright retinal field."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    center = (width // 2, height // 2)
    radius = min(height, width) // 3

    cv2.circle(
        image,
        center,
        radius,
        (120, 80, 50),
        thickness=-1,
    )

    return image


def test_preprocessing_pipeline_output_contract():
    """Pipeline should return a normalized 512x512 float32 image."""
    image = make_synthetic_retina()

    pipeline = PreprocessingPipeline(
        target_size=(512, 512)
    )

    output = pipeline.process(image)

    assert output.shape == (512, 512, 3)
    assert output.dtype == np.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_preprocessing_pipeline_does_not_modify_input():
    """Pipeline must not mutate the input image."""
    image = make_synthetic_retina()
    original = image.copy()

    pipeline = PreprocessingPipeline(
        target_size=(512, 512)
    )

    pipeline.process(image)

    np.testing.assert_array_equal(image, original)


def test_crop_retina_reduces_dark_border():
    """Retinal cropping should reduce the surrounding dark border."""
    image = make_synthetic_retina(
        height=600,
        width=800,
    )

    cropped = crop_retina_circle(image)

    assert cropped.ndim == 3
    assert cropped.shape[2] == 3

    assert cropped.shape[0] <= image.shape[0]
    assert cropped.shape[1] <= image.shape[1]

    assert (
        cropped.shape[0] < image.shape[0]
        or cropped.shape[1] < image.shape[1]
    )


def test_crop_retina_handles_empty_foreground():
    """A completely dark image should not cause a crash."""
    image = np.zeros(
        (500, 500, 3),
        dtype=np.uint8,
    )

    cropped = crop_retina_circle(image)

    assert cropped.shape == image.shape


def test_crop_retina_rejects_invalid_shape():
    """Crop should reject non-RGB images."""
    image = np.zeros(
        (500, 500),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):
        crop_retina_circle(image)


def test_illumination_normalization_output():
    """Illumination normalization should preserve image shape and dtype."""
    image = make_synthetic_retina()

    output = normalize_illumination(
        image,
        sigma=10.0,
    )

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert output.min() >= 0
    assert output.max() <= 255


def test_illumination_rejects_invalid_sigma():
    """Sigma must be positive."""
    image = make_synthetic_retina()

    with pytest.raises(ValueError):
        normalize_illumination(
            image,
            sigma=0,
        )


def test_clahe_output():
    """CLAHE should preserve RGB image shape and dtype."""
    image = make_synthetic_retina()

    output = apply_clahe(
        image,
        clip_limit=2.0,
    )

    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert output.min() >= 0
    assert output.max() <= 255


def test_clahe_rejects_invalid_clip_limit():
    """CLAHE clip limit must be positive."""
    image = make_synthetic_retina()

    with pytest.raises(ValueError):
        apply_clahe(
            image,
            clip_limit=0,
        )


@pytest.mark.parametrize(
    "size",
    [
        (474, 358),
        (1050, 1050),
        (2416, 1736),
        (4288, 2848),
    ],
)
def test_pipeline_handles_heterogeneous_sizes(size):
    """Pipeline should handle representative APTOS resolutions."""
    width, height = size

    image = make_synthetic_retina(
        height=height,
        width=width,
    )

    pipeline = PreprocessingPipeline(
        target_size=(512, 512)
    )

    output = pipeline.process(image)

    assert output.shape == (512, 512, 3)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_pipeline_with_clahe():
    """Optional CLAHE branch should produce valid model input."""
    image = make_synthetic_retina()

    pipeline = PreprocessingPipeline(
        target_size=(512, 512),
        use_clahe=True,
    )

    output = pipeline.process(image)

    assert output.shape == (512, 512, 3)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_pipeline_with_illumination_normalization():
    """Optional illumination branch should produce valid model input."""
    image = make_synthetic_retina()

    pipeline = PreprocessingPipeline(
        target_size=(512, 512),
        use_illumination_normalization=True,
    )

    output = pipeline.process(image)

    assert output.shape == (512, 512, 3)
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
