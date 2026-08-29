from pathlib import Path

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_preprocess_image():
    image = Image.open(FIXTURE_DIR / "test.png")

    result = preprocess_image(image)

    assert result.shape == (1, 48, 48, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0
