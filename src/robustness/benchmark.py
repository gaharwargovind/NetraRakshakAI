from typing import Dict, Any, List
from PIL import Image
from src.robustness.blur import apply_gaussian_blur
from src.robustness.noise import apply_gaussian_noise
from src.robustness.brightness import adjust_brightness, adjust_contrast
from src.robustness.compression import apply_jpeg_compression

PERTURBATION_SUITE: Dict[str, List[Dict[str, Any]]] = {
    "gaussian_blur": [
        {"severity": 1, "desc": "Subtle blur (k=3, s=0.5)", "params": {"kernel_size": 3, "sigma": 0.5}},
        {"severity": 2, "desc": "Mild defocus (k=5, s=1.0)", "params": {"kernel_size": 5, "sigma": 1.0}},
        {"severity": 3, "desc": "Moderate defocus (k=9, s=2.0)", "params": {"kernel_size": 9, "sigma": 2.0}},
        {"severity": 4, "desc": "Severe blur (k=15, s=3.5)", "params": {"kernel_size": 15, "sigma": 3.5}},
        {"severity": 5, "desc": "Extreme blur (k=21, s=5.0)", "params": {"kernel_size": 21, "sigma": 5.0}},
    ],
    "gaussian_noise": [
        {"severity": 1, "desc": "Low sensor noise (sigma=5)", "params": {"sigma": 5.0}},
        {"severity": 2, "desc": "Mild sensor noise (sigma=15)", "params": {"sigma": 15.0}},
        {"severity": 3, "desc": "Moderate sensor noise (sigma=25)", "params": {"sigma": 25.0}},
        {"severity": 4, "desc": "High sensor noise (sigma=40)", "params": {"sigma": 40.0}},
        {"severity": 5, "desc": "Severe sensor noise (sigma=60)", "params": {"sigma": 60.0}},
    ],
    "brightness": [
        {"severity": 1, "desc": "Severe underexposure (-60%)", "params": {"factor": 0.4}},
        {"severity": 2, "desc": "Moderate underexposure (-30%)", "params": {"factor": 0.7}},
        {"severity": 3, "desc": "Moderate overexposure (+30%)", "params": {"factor": 1.3}},
        {"severity": 4, "desc": "Severe overexposure (+60%)", "params": {"factor": 1.6}},
    ],
    "contrast": [
        {"severity": 1, "desc": "Severe contrast loss (-60%)", "params": {"factor": 0.4}},
        {"severity": 2, "desc": "Moderate contrast loss (-30%)", "params": {"factor": 0.7}},
        {"severity": 3, "desc": "Moderate contrast increase (+30%)", "params": {"factor": 1.3}},
        {"severity": 4, "desc": "Severe contrast increase (+60%)", "params": {"factor": 1.6}},
    ],
    "jpeg_compression": [
        {"severity": 1, "desc": "Light compression (Q=80)", "params": {"quality": 80}},
        {"severity": 2, "desc": "Moderate compression (Q=60)", "params": {"quality": 60}},
        {"severity": 3, "desc": "Aggressive compression (Q=40)", "params": {"quality": 40}},
        {"severity": 4, "desc": "Severe compression (Q=20)", "params": {"quality": 20}},
        {"severity": 5, "desc": "Extreme compression (Q=10)", "params": {"quality": 10}},
    ]
}

def apply_perturbation(family: str, img: Image.Image, params: Dict[str, Any], seed: int = 42) -> Image.Image:
    if family == "gaussian_blur":
        return apply_gaussian_blur(img, params["kernel_size"], params["sigma"])
    elif family == "gaussian_noise":
        return apply_gaussian_noise(img, params["sigma"], seed=seed)
    elif family == "brightness":
        return adjust_brightness(img, params["factor"])
    elif family == "contrast":
        return adjust_contrast(img, params["factor"])
    elif family == "jpeg_compression":
        return apply_jpeg_compression(img, params["quality"])
    else:
        raise ValueError(f"Unknown perturbation family: {family}")
