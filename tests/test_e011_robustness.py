import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent

from src.robustness.blur import apply_gaussian_blur
from src.robustness.noise import apply_gaussian_noise
from src.robustness.brightness import adjust_brightness, adjust_contrast
from src.robustness.compression import apply_jpeg_compression

@pytest.fixture
def sample_image():
    # Deterministic test pattern
    arr = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
    return Image.fromarray(arr)

def test_no_mutation_of_source(sample_image):
    original_bytes = sample_image.tobytes()
    _ = apply_gaussian_blur(sample_image, kernel_size=5, sigma=1.0)
    _ = apply_gaussian_noise(sample_image, sigma=20.0, seed=42)
    _ = adjust_brightness(sample_image, factor=0.5)
    _ = adjust_contrast(sample_image, factor=0.5)
    _ = apply_jpeg_compression(sample_image, quality=50)
    assert sample_image.tobytes() == original_bytes

def test_noise_reproducibility(sample_image):
    n1 = apply_gaussian_noise(sample_image, sigma=15.0, seed=123)
    n2 = apply_gaussian_noise(sample_image, sigma=15.0, seed=123)
    n3 = apply_gaussian_noise(sample_image, sigma=15.0, seed=999)
    assert np.array_equal(np.array(n1), np.array(n2))
    assert not np.array_equal(np.array(n1), np.array(n3))

def test_valid_pixel_ranges(sample_image):
    perturbed = [
        apply_gaussian_blur(sample_image, 7, 2.0),
        apply_gaussian_noise(sample_image, 50.0, seed=42),
        adjust_brightness(sample_image, 1.8),
        adjust_contrast(sample_image, 0.3),
        apply_jpeg_compression(sample_image, 10),
    ]
    for img in perturbed:
        arr = np.array(img)
        assert arr.min() >= 0
        assert arr.max() <= 255
        assert arr.shape == (64, 64, 3)

def test_blur_severity_monotonicity(sample_image):
    b1 = np.array(apply_gaussian_blur(sample_image, 3, 0.5), dtype=float)
    b2 = np.array(apply_gaussian_blur(sample_image, 15, 3.5), dtype=float)
    orig = np.array(sample_image, dtype=float)
    diff1 = np.mean(np.abs(b1 - orig))
    diff2 = np.mean(np.abs(b2 - orig))
    assert diff2 > diff1

def test_e007_checkpoint_integrity():
    ckpt_path = ROOT_DIR / "models/checkpoints/E007_best_model.pt"
    assert ckpt_path.is_file()
    # Verified E007 sha
    expected_sha = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"
    current_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert current_sha == expected_sha
