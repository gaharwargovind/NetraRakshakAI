"""Unit and invariant tests for Phase E014 Saliency and Counterfactual Audit."""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, 5)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(pooled)


class MockDREfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockBackbone()

    def forward(self, x):
        return self.model(x)


class GradCAMTestEngine:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, output.detach()

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def test_gradcam_output_contract():
    model = MockDREfficientNet()
    model.eval()
    target_layer = model.model.features[-2]
    cam_engine = GradCAMTestEngine(model, target_layer)

    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)
    cam, logits = cam_engine.generate(dummy_input)

    assert cam.shape == (256, 256)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0
    assert logits.shape == (1, 5)
    cam_engine.remove_hooks()


def test_counterfactual_masking_contract():
    img_np = np.ones((512, 512, 3), dtype=np.uint8) * 100
    cam = np.zeros((512, 512), dtype=np.float32)
    cam[100:200, 100:200] = 0.8

    threshold = 0.5
    salient_masked = img_np.copy()
    salient_masked[cam >= threshold] = 0

    bg_masked = img_np.copy()
    bg_masked[cam < threshold] = 0

    assert salient_masked.shape == (512, 512, 3)
    assert bg_masked.shape == (512, 512, 3)
    assert (salient_masked[150, 150] == 0).all()
    assert (salient_masked[50, 50] == 100).all()
    assert (bg_masked[150, 150] == 100).all()
    assert (bg_masked[50, 50] == 0).all()


def test_saliency_energy_partition():
    cam = np.random.uniform(0, 1, (512, 512)).astype(np.float32)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:400, 100:400] = 1

    total_energy = float(np.sum(cam))
    fg_energy = float(np.sum(cam[mask > 0]))
    bg_energy = float(np.sum(cam[mask == 0]))

    assert np.isclose(fg_energy + bg_energy, total_energy, atol=1e-5)
    fraction_fg = fg_energy / total_energy
    fraction_bg = bg_energy / total_energy
    assert 0.0 <= fraction_fg <= 1.0
    assert 0.0 <= fraction_bg <= 1.0
    assert np.isclose(fraction_fg + fraction_bg, 1.0, atol=1e-5)


def test_model_parameter_freeze_invariance():
    model = MockDREfficientNet()
    orig_params = [p.clone() for p in model.parameters()]
    target_layer = model.model.features[-2]
    cam_engine = GradCAMTestEngine(model, target_layer)

    dummy_input = torch.randn(1, 3, 64, 64, requires_grad=True)
    _ = cam_engine.generate(dummy_input)

    for p_orig, p_curr in zip(orig_params, model.parameters()):
        assert torch.equal(p_orig, p_curr)
    cam_engine.remove_hooks()
