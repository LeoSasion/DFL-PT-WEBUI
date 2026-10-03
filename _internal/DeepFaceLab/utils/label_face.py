import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.imagelib.text import _get_pil_font


def label_face_filename(face, filename):
    """Return a labeled BGR float image, including Unicode filenames."""
    pixels = np.clip(np.asarray(face) * 255.0, 0, 255).round().astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    font = _get_pil_font(None, 15)
    ImageDraw.Draw(image).text((5, max(0, pixels.shape[0] - 20)), str(filename),
                              fill=(255, 128, 0), font=font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
