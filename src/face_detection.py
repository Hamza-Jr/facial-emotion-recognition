import cv2
import numpy as np
from PIL import Image

from src.config import YUNET_MODEL_PATH


detector = cv2.FaceDetectorYN.create(
    str(YUNET_MODEL_PATH),
    "",
    (400, 400),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=1,
)


def detect_and_crop_face(image: Image.Image) -> Image.Image | None:
    """Detect the primary face and return its original-resolution crop."""

    image_rgb = np.asarray(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    height, width = image_bgr.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(image_bgr)

    if faces is None:
        return None

    x, y, w, h = faces[0][:4].astype(int)
    padding = int(max(w, h) * 0.30)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    return Image.fromarray(image_rgb[y1:y2, x1:x2])
