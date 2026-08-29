import numpy as np
from PIL import Image
from src.config import IMAGE_SIZE, NORMALIZATION_FACTOR


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess an image for model inference."""

    # 1. Convert to grayscale and resize
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE)

    # 2. Normalize and format for TensorFlow/Keras
    image_array = np.asarray(image, dtype=np.float32)
    image_array /= NORMALIZATION_FACTOR

    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
