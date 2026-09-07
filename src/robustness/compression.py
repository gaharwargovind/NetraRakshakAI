import io
from PIL import Image

def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    quality = max(1, min(100, int(quality)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
