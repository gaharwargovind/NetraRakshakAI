from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetGradCAM:
    """
    Grad-CAM implementation targeting convolutional stages in EfficientNet-B0.
    Calculates gradient-weighted activation maps for specified target classes.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        # By default target the last conv block of features
        if target_layer is None:
            if hasattr(model, "model") and hasattr(model.model, "features"):
                self.target_layer = model.model.features[-1]
            elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
                self.target_layer = model.backbone.features[-1]
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
            else:
                raise ValueError("Target layer could not be automatically identified.")
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, torch.Tensor]:
        """
        Generates Grad-CAM heatmap normalized to [0, 1].
        input_tensor: Shape (1, 3, H, W)
        """
        # Grad-CAM requires an autograd graph even though the E007
        # model weights remain frozen for inference.
        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(input_tensor)

        probs = F.softmax(logits, dim=-1).squeeze(0)

        if target_class is None:
            target_class = int(torch.argmax(probs).item())

        target_score = logits[0, target_class]
        target_score.backward(retain_graph=False)

        # Global average pooling of gradients
        # Gradients shape: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap, target_class, probs

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
