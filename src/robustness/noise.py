import numpy as np
from PIL import Image

def apply_gaussian_noise(img: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    if sigma <= 0:
        return img.copy()
    arr = np.array(img, dtype=np.float32)
    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(noisy_arr)
