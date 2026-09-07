import cv2
import numpy as np
from PIL import Image

def apply_gaussian_blur(img: Image.Image, kernel_size: int, sigma: float) -> Image.Image:
    if kernel_size <= 1 or sigma <= 0:
        return img.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    arr = np.array(img)
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred)
