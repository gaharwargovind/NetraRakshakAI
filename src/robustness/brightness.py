from PIL import Image, ImageEnhance

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)
