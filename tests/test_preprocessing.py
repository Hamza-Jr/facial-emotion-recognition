import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image


def test_preprocess_image():
    image = Image.new("RGB", (200, 200), color="white")

    result = preprocess_image(image)

    assert result.shape == (1, 48, 48, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0
