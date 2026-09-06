import hashlib
from pathlib import Path
import numpy as np
import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent

from src.models.efficientnet import DREfficientNet
from src.explainability.gradcam import EfficientNetGradCAM
from src.explainability.overlays import generate_cam_overlay, extract_retinal_mask, compute_attention_distribution
from src.explainability.lesion_comparison import compute_saliency_mask_metrics

def test_gradcam_output_dimensions_and_bounds():
    """Grad-CAM must output a 2D array matching input spatial dimensions with values in [0, 1]."""
    model = DREfficientNet(backbone_name="efficientnet_b0", num_classes=5, pretrained=False)
    gradcam = EfficientNetGradCAM(model, target_layer=model.model.features[-1])

    fake_input = torch.randn(1, 3, 512, 512)
    heatmap, target_cls, probs = gradcam.generate(fake_input, target_class=2)
    gradcam.remove_hooks()

    assert heatmap.shape == (512, 512)
    assert np.all(np.isfinite(heatmap))
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert target_cls == 2

def test_overlay_generation():
    """Overlay must produce uint8 RGB array matching image dimensions."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    heatmap = np.ones((512, 512), dtype=np.float32) * 0.5
    overlay = generate_cam_overlay(img, heatmap)

    assert overlay.shape == (512, 512, 3)
    assert overlay.dtype == np.uint8

def test_attention_distribution():
    """Retinal mask and energy calculation must partition energy to sum to 1.0."""
    cv2_circle = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2_circle[100:400, 100:400] = 100
    heatmap = np.ones((512, 512), dtype=np.float32)

    ret_ratio, border_ratio = compute_attention_distribution(heatmap, cv2_circle)
    assert np.isclose(ret_ratio + border_ratio, 1.0, atol=1e-4)

def test_e007_checkpoint_integrity_unmodified():
    """E007 checkpoint must not be mutated by E008 audit."""
    ckpt_path = ROOT / "models/checkpoints/E007_best_model.pt"
    initial_file = ROOT / "experiments/E008_gradcam/.e007_initial_sha"
    if initial_file.is_file() and ckpt_path.is_file():
        initial_sha = initial_file.read_text().strip()
        current_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        assert current_sha == initial_sha

def test_test_csv_not_referenced_in_e008():
    """Confirm held-out test partition is strictly unreferenced."""
    script_path = ROOT / "scripts/evaluate_e008_explainability.py"
    if script_path.is_file():
        script = script_path.read_text()
        assert "data/processed/aptos/test.csv" not in script
