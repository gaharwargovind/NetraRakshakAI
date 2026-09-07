"""Deterministic fundus image quality assessment (IQA) feature extraction."""

from dataclasses import dataclass
from typing import Dict, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class IQAMetrics:
    fov_coverage: float
    laplacian_variance: float
    edge_density: float
    mean_intensity: float
    dark_fraction: float
    bright_fraction: float
    illumination_uniformity: float
    intensity_std: float
    percentile_spread_90: float
    noise_mad: float


class FundusIQAExtractor:
    """Extracts deterministic quality metrics constrained to retinal foreground."""

    def __init__(
        self,
        min_retina_intensity: int = 15,
        dark_thresh: int = 15,
        bright_thresh: int = 240,
        grid_dim: int = 4,
    ):
        self.min_retina_intensity = min_retina_intensity
        self.dark_thresh = dark_thresh
        self.bright_thresh = bright_thresh
        self.grid_dim = grid_dim

    def extract_retinal_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        _, mask = cv2.threshold(gray, self.min_retina_intensity, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def extract_metrics(self, img_bgr: np.ndarray) -> IQAMetrics:
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image must be a valid non-empty numpy array.")

        h, w = img_bgr.shape[:2]
        total_pixels = float(h * w)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        mask = self.extract_retinal_mask(img_bgr)
        foreground_pixels = int(np.count_nonzero(mask))

        if foreground_pixels < 100:
            return IQAMetrics(
                fov_coverage=0.0,
                laplacian_variance=0.0,
                edge_density=0.0,
                mean_intensity=float(np.mean(gray)),
                dark_fraction=1.0,
                bright_fraction=0.0,
                illumination_uniformity=0.0,
                intensity_std=0.0,
                percentile_spread_90=0.0,
                noise_mad=0.0,
            )

        fg_gray = gray[mask > 0]
        fov_coverage = float(foreground_pixels / total_pixels)

        # 1. Focus / Blur
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        fg_lap = lap[mask > 0]
        lap_var = float(np.var(fg_lap))

        edges = cv2.Canny(gray, 30, 100)
        fg_edges = edges[mask > 0]
        edge_density = float(np.count_nonzero(fg_edges) / foreground_pixels)

        # 2. Illumination & Uniformity
        mean_intensity = float(np.mean(fg_gray))
        dark_fraction = float(np.count_nonzero(fg_gray < self.dark_thresh) / foreground_pixels)
        bright_fraction = float(np.count_nonzero(fg_gray > self.bright_thresh) / foreground_pixels)

        cell_means = []
        cell_h, cell_w = h // self.grid_dim, w // self.grid_dim
        for r in range(self.grid_dim):
            for c in range(self.grid_dim):
                sub_mask = mask[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                if np.count_nonzero(sub_mask) > (0.1 * cell_h * cell_w):
                    sub_gray = gray[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                    cell_means.append(np.mean(sub_gray[sub_mask > 0]))

        if len(cell_means) > 1 and np.mean(cell_means) > 1e-3:
            grid_cv = float(np.std(cell_means) / np.mean(cell_means))
            illumination_uniformity = float(max(0.0, 1.0 - grid_cv))
        else:
            illumination_uniformity = 0.0

        # 3. Contrast
        intensity_std = float(np.std(fg_gray))
        p95 = float(np.percentile(fg_gray, 95))
        p05 = float(np.percentile(fg_gray, 5))
        p90_spread = float(max(0.0, p95 - p05))

        # 4. Noise Proxy (Donoho MAD on median filter residual)
        blurred = cv2.medianBlur(gray, 3)
        residual = gray.astype(np.float32) - blurred.astype(np.float32)
        fg_res = residual[mask > 0]
        mad = float(np.median(np.abs(fg_res - np.median(fg_res))))
        noise_mad = float(mad / 0.6745)

        return IQAMetrics(
            fov_coverage=round(fov_coverage, 4),
            laplacian_variance=round(lap_var, 2),
            edge_density=round(edge_density, 4),
            mean_intensity=round(mean_intensity, 2),
            dark_fraction=round(dark_fraction, 4),
            bright_fraction=round(bright_fraction, 4),
            illumination_uniformity=round(illumination_uniformity, 4),
            intensity_std=round(intensity_std, 2),
            percentile_spread_90=round(p90_spread, 2),
            noise_mad=round(noise_mad, 2),
        )
