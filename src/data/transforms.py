"""Model-facing transforms for E001 Baseline, E002 CLAHE, and E004 Augmentation."""

import random
from typing import Any, Dict, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from src.preprocessing.clahe import apply_clahe
from src.preprocessing.crop_retina import crop_retina_circle


class E001BaselineTransform:
    """
    Deterministic baseline transform pipeline:
    Raw image -> Retinal Crop -> Bilinear Resize to 512x512 -> Float32 [0, 1] -> PyTorch Tensor.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)) -> None:
        self.target_size = target_size

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"), dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img_np = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img_np = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                img_np = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        cropped = crop_retina_circle(img_np)
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        return tensor


class E002CLAHENormalizeTransform:
    """
    E002 CLAHE experimental transform pipeline.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.target_size = target_size
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"), dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img_np = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img_np = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                img_np = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        cropped = crop_retina_circle(img_np)
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_LINEAR)
        enhanced = apply_clahe(
            resized,
            clip_limit=self.clip_limit,
            tile_grid_size=self.tile_grid_size,
        )
        normalized = enhanced.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        return tensor


class E004AugmentationTransform:
    """
    Conservative, clinically plausible training augmentation for fundus images:
    1. Retinal circular crop
    2. Bilinear resize to target_size (512, 512)
    3. Random Horizontal Flip (p = 0.5)
    4. Random Rotation (uniform within [-10, +10] degrees, p = 0.7)
    5. Random Affine Translation (dx, dy in [-2%, +2%]) and Scale (uniform in [0.96, 1.04], p = 0.5)
    6. Mild Color Jitter (Brightness in [0.95, 1.05], Contrast in [0.95, 1.05], p = 0.5)
    7. Float32 normalization to [0.0, 1.0]
    8. Transpose (H, W, C) -> (C, H, W) and convert to Tensor.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        hflip_p: float = 0.5,
        rotation_p: float = 0.7,
        max_rotation_deg: float = 10.0,
        affine_p: float = 0.5,
        scale_range: Tuple[float, float] = (0.96, 1.04),
        translate_pct: float = 0.02,
        jitter_p: float = 0.5,
        brightness_range: Tuple[float, float] = (0.95, 1.05),
        contrast_range: Tuple[float, float] = (0.95, 1.05),
    ) -> None:
        self.target_size = target_size
        self.hflip_p = hflip_p
        self.rotation_p = rotation_p
        self.max_rotation_deg = max_rotation_deg
        self.affine_p = affine_p
        self.scale_range = scale_range
        self.translate_pct = translate_pct
        self.jitter_p = jitter_p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"), dtype=np.uint8)
        elif isinstance(img, np.ndarray):
            if img.ndim == 2:
                img_np = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img_np = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                img_np = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        # Step 1: Retinal circular crop
        cropped = crop_retina_circle(img_np)

        # Step 2: Bilinear resize to 512x512
        h_t, w_t = self.target_size
        resized = cv2.resize(cropped, (w_t, h_t), interpolation=cv2.INTER_LINEAR)

        # Step 3: Horizontal Flip
        if random.random() < self.hflip_p:
            resized = cv2.flip(resized, 1)

        # Step 4 & 5: Geometric Affine (Rotation + Scale + Translation)
        do_rotation = random.random() < self.rotation_p
        do_affine = random.random() < self.affine_p

        if do_rotation or do_affine:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg) if do_rotation else 0.0
            scale = random.uniform(self.scale_range[0], self.scale_range[1]) if do_affine else 1.0
            tx = random.uniform(-self.translate_pct, self.translate_pct) * w_t if do_affine else 0.0
            ty = random.uniform(-self.translate_pct, self.translate_pct) * h_t if do_affine else 0.0

            center = (w_t / 2.0, h_t / 2.0)
            mat = cv2.getRotationMatrix2D(center, angle, scale)
            mat[0, 2] += tx
            mat[1, 2] += ty
            resized = cv2.warpAffine(
                resized,
                mat,
                (w_t, h_t),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        # Step 6: Mild Color Jitter (Brightness & Contrast)
        if random.random() < self.jitter_p:
            alpha = random.uniform(self.contrast_range[0], self.contrast_range[1])
            beta_mult = random.uniform(self.brightness_range[0], self.brightness_range[1])
            arr_float = resized.astype(np.float32) * beta_mult
            arr_float = (arr_float - 128.0) * alpha + 128.0
            resized = np.clip(arr_float, 0, 255).astype(np.uint8)

        # Step 7: Float32 normalization to [0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0

        # Step 8: Convert to PyTorch Tensor (C, H, W)
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        return tensor


def get_training_transforms(config: Dict[str, Any]) -> E001BaselineTransform:
    image_size = config.get("data", {}).get("image_size", [512, 512])
    return E001BaselineTransform(target_size=(image_size[0], image_size[1]))


def get_validation_transforms(config: Dict[str, Any]) -> E001BaselineTransform:
    image_size = config.get("data", {}).get("image_size", [512, 512])
    return E001BaselineTransform(target_size=(image_size[0], image_size[1]))