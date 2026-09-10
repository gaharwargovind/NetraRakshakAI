"""Production-grade deterministic inference pipeline for NetraRakshakAI."""

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.quality import FundusIQAExtractor
from src.data.transforms import E001BaselineTransform
from src.explainability.gradcam import EfficientNetGradCAM
from src.inference.quality_gate import FundusQualityGate, GateStatus
from src.models.efficientnet import DREfficientNet

logger = logging.getLogger("inference_pipeline")
EXPECTED_E007_SHA = "a61710e11557bb7d1be60ed488e5bdf5b88c92d16c76441513bbfa4d8b94cc3c"
FROZEN_TEMPERATURE = 0.7785
LOW_CONFIDENCE_THRESHOLD = 0.60  # Documented engineering placeholder; not clinically validated
DISABLE_GRADCAM = os.getenv("NETRA_DISABLE_GRADCAM", "").strip().lower() in {"1", "true", "yes", "on"}


def verify_checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class ScreeningPredictor:
    """Consolidated end-to-end inference engine for diabetic retinopathy screening."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None
    ) -> None:
        self.config = config or {}
        repo_root = Path.cwd()
        default_ckpt = repo_root / "models/checkpoints/E007_best_model.pt"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_ckpt

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.checkpoint_sha = verify_checkpoint_sha256(self.checkpoint_path)
        if self.checkpoint_sha != EXPECTED_E007_SHA:
            raise ValueError(
                f"Integrity check failed. Expected {EXPECTED_E007_SHA}, got {self.checkpoint_sha}"
            )

        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load frozen E007 model
        self.model = DREfficientNet(
            backbone_name="efficientnet_b0",
            num_classes=5,
            pretrained=False,
            dropout_rate=0.2
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.quality_gate = FundusQualityGate()
        self.transform = E001BaselineTransform(target_size=(512, 512))
        self.temperature = FROZEN_TEMPERATURE

        # Initialize Grad-CAM targeting features[-1]
        try:
            self.gradcam = EfficientNetGradCAM(self.model, target_layer=self.model.model.features[-1])
        except Exception as e:
            logger.warning("Grad-CAM initialization deferred or failed: %s", e)
            self.gradcam = None

    def _sanitize_input(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Tuple[np.ndarray, Image.Image]:
        """Validates and standardizes input to BGR numpy array and RGB PIL image."""
        if image is None:
            raise ValueError("Input image cannot be None.")

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {path}")
            pil_img = Image.open(path).convert("RGB")
            rgb_np = np.array(pil_img)
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            return bgr_np, pil_img

        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            rgb_np = np.array(pil_img)
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            return bgr_np, pil_img

        if isinstance(image, np.ndarray):
            if np.isnan(image).any() or np.isinf(image).any():
                raise ValueError("Image array contains invalid NaN or Inf values.")

            if image.ndim == 2:  # Grayscale
                bgr_np = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                rgb_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return bgr_np, Image.fromarray(rgb_np)

            if image.ndim == 3:
                channels = image.shape[2]
                if channels == 1:
                    sq = image.squeeze(2)
                    bgr_np = cv2.cvtColor(sq, cv2.COLOR_GRAY2BGR)
                    rgb_np = cv2.cvtColor(sq, cv2.COLOR_GRAY2RGB)
                    return bgr_np, Image.fromarray(rgb_np)
                elif channels == 3:
                    # Assume BGR if coming from OpenCV array convention
                    bgr_np = image.copy()
                    rgb_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    return bgr_np, Image.fromarray(rgb_np)
                elif channels == 4:
                    bgr_np = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    rgb_np = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    return bgr_np, Image.fromarray(rgb_np)
                else:
                    raise ValueError(f"Unsupported channel dimension: {channels}")

            raise ValueError(f"Unsupported array shape: {image.shape}")

        raise TypeError(f"Unsupported input type: {type(image)}")

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, str, Path],
        image_id: Optional[str] = None,
        generate_saliency: bool = True
    ) -> Dict[str, Any]:
        """Runs the deterministic screening pipeline with quality triage."""
        img_id = image_id or f"img_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:21]}"

        # Step 1: Input Validation
        try:
            bgr_img, pil_img = self._sanitize_input(image)
        except Exception as e:
            return self._build_error_response(img_id, str(e))

        # Step 2: Deterministic IQA Gate Evaluation
        gate_result = self.quality_gate.evaluate(bgr_img)
        m = gate_result.metrics
        metrics_dict = {
            "fov_coverage": round(m.fov_coverage, 4) if m else 0.0,
            "laplacian_variance": round(m.laplacian_variance, 2) if m else 0.0,
            "edge_density": round(m.edge_density, 6) if m else 0.0,
            "mean_intensity": round(m.mean_intensity, 2) if m else 0.0,
            "dark_fraction": round(m.dark_fraction, 4) if m else 0.0,
            "bright_fraction": round(m.bright_fraction, 4) if m else 0.0,
            "percentile_spread_90": round(m.percentile_spread_90, 2) if m else 0.0,
            "noise_mad": round(m.noise_mad, 4) if m else 0.0
        }

        # Step 3: Hard Quality Gate Intercept
        if gate_result.status == GateStatus.FAIL:
            return {
                "image_id": img_id,
                "quality": {
                    "status": "FAIL",
                    "metrics": metrics_dict,
                    "failed_checks": gate_result.rejection_reasons,
                    "message": "Image quality insufficient for automated grading. Recapture recommended."
                },
                "classification": None,
                "calibration": {
                    "temperature": self.temperature,
                    "calibrated": False,
                    "note": "Skipped due to upstream IQA failure."
                },
                "explanation": {
                    "gradcam_available": False,
                    "target_layer": "model.model.features[-1]",
                    "status": "skipped",
                    "heatmap": None,
                    "overlay": None
                },
                "recommendation": {
                    "action": "RECAPTURE_OR_HUMAN_REVIEW",
                    "reason": f"Quality triage gate rejected image: {'; '.join(gate_result.rejection_reasons)}"
                },
                "metadata": self._build_metadata()
            }

        # Step 4: Canonical E007 Inference
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_logits = self.model(input_tensor)
            calibrated_logits = raw_logits / self.temperature
            probs_t = F.softmax(calibrated_logits, dim=-1).squeeze(0)
            probs = probs_t.cpu().numpy().tolist()

        predicted_grade = int(np.argmax(probs))
        referable = bool(predicted_grade >= 2)
        referable_prob = float(np.sum(probs[2:]))
        confidence = float(probs[predicted_grade])

        # Step 5: Grad-CAM Saliency Generation with Graceful Fallback
        cam_heatmap = None
        cam_overlay = None
        gradcam_status = "disabled"

        if generate_saliency and self.gradcam and not DISABLE_GRADCAM:
            try:
                # Grad-CAM requires an autograd graph for the attribution backward pass.
                # Keep E007 weights frozen; enable gradients only for this explanation step.
                input_tensor_grad = (
                    input_tensor.clone()
                    .detach()
                    .requires_grad_(True)
                )

                with torch.enable_grad():
                    heatmap, _, _ = self.gradcam.generate(
                        input_tensor_grad,
                        target_class=predicted_grade
                    )

                cam_heatmap = heatmap

                # Build visual overlay
                inv_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                inv_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
                unnorm = (input_tensor[0] * inv_std + inv_mean).clamp(0, 1)
                rgb_512 = (unnorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                colored_heat = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                colored_heat = cv2.cvtColor(colored_heat, cv2.COLOR_BGR2RGB)
                cam_overlay = np.uint8(0.45 * colored_heat + 0.55 * rgb_512)
                gradcam_status = "generated"
            except Exception as e:
                logger.warning("Grad-CAM generation failed gracefully: %s", e)
                gradcam_status = "unavailable"

        if generate_saliency and DISABLE_GRADCAM:
            gradcam_status = "disabled"

        # Step 6: Escalation and Recommendation Triage
        if referable:
            action = "SPECIALIST_REFERRAL"
            reason = f"Referable diabetic retinopathy detected (ICDR Grade {predicted_grade} >= 2)."
        elif confidence < LOW_CONFIDENCE_THRESHOLD:
            action = "HUMAN_REVIEW"
            reason = f"Low model confidence ({confidence*100:.1f}% < {LOW_CONFIDENCE_THRESHOLD*100:.0f}%). Clinical review required."
        else:
            action = "ROUTINE_MONITORING"
            reason = f"Non-referable diabetic retinopathy (ICDR Grade {predicted_grade}). Subject to clinical screening protocol."

        return {
            "image_id": img_id,
            "quality": {
                "status": "PASS",
                "metrics": metrics_dict,
                "failed_checks": [],
                "message": "Image quality acceptable for AI screening triage."
            },
            "classification": {
                "predicted_grade": predicted_grade,
                "class_probabilities": [round(p, 4) for p in probs],
                "referable": referable,
                "referable_probability": round(referable_prob, 4),
                "confidence": round(confidence, 4)
            },
            "calibration": {
                "temperature": self.temperature,
                "calibrated": True,
                "method": "Frozen development calibration from APTOS validation."
            },
            "explanation": {
                "gradcam_available": (gradcam_status == "generated"),
                "target_layer": "model.model.features[-1]",
                "status": gradcam_status,
                "heatmap": cam_heatmap,
                "overlay": cam_overlay
            },
            "recommendation": {
                "action": action,
                "reason": reason
            },
            "metadata": self._build_metadata()
        }

    def _build_metadata(self) -> Dict[str, Any]:
        return {
            "model_version": "E007",
            "model_checkpoint_sha256": self.checkpoint_sha,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "1.0.0-E015",
            "scientific_mandate": "AI-assisted diabetic retinopathy screening; clinical validation of gradability and diagnostic adjudication remain necessary."
        }

    def _build_error_response(self, image_id: str, error_msg: str) -> Dict[str, Any]:
        return {
            "image_id": image_id,
            "quality": {
                "status": "FAIL",
                "metrics": {},
                "failed_checks": [f"Input validation error: {error_msg}"],
                "message": f"Input validation rejected image: {error_msg}"
            },
            "classification": None,
            "calibration": {
                "temperature": self.temperature,
                "calibrated": False,
                "note": "Halted prior to inference."
            },
            "explanation": {
                "gradcam_available": False,
                "target_layer": "model.model.features[-1]",
                "status": "unavailable",
                "heatmap": None,
                "overlay": None
            },
            "recommendation": {
                "action": "RECAPTURE_OR_HUMAN_REVIEW",
                "reason": f"Input data rejected: {error_msg}"
            },
            "metadata": self._build_metadata()
        }
