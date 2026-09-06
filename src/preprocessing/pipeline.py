"""End-to-end retinal image preprocessing pipeline."""

from typing import Tuple

import cv2
import numpy as np

from .clahe import apply_clahe
from .crop_retina import crop_retina_circle
from .illumination import normalize_illumination


class PreprocessingPipeline:
    """
    Standardize heterogeneous fundus images.

    Default pipeline:
        crop -> resize -> normalize

    Optional experimental transformations:
        illumination normalization
        CLAHE
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        use_illumination_normalization: bool = False,
        use_clahe: bool = False,
    ) -> None:
        self.target_size = target_size
        self.use_illumination_normalization = (
            use_illumination_normalization
        )
        self.use_clahe = use_clahe

        if len(target_size) != 2:
            raise ValueError(
                "target_size must contain (height, width)"
            )

        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(
                "target dimensions must be positive"
            )

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process an RGB image into a normalized 512x512 representation.

        Returns:
            Float32 RGB array with values in [0, 1].
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "image must have shape (H, W, 3)"
            )

        working = image.copy()

        # 1. Detect and crop the retinal field.
        working = crop_retina_circle(working)

        # 2. Resize before optional image normalization.
        height, width = self.target_size

        working = cv2.resize(
            working,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

        # 3. Optional illumination normalization.
        # Applied at standardized resolution so sigma has
        # consistent meaning across heterogeneous source sizes.
        if self.use_illumination_normalization:
            working = normalize_illumination(
                working,
                sigma=10.0,
            )

        # 4. Optional CLAHE.
        if self.use_clahe:
            working = apply_clahe(working)

        # 5. Convert uint8 RGB -> float32 [0, 1].
        output = working.astype(np.float32) / 255.0

        return output
