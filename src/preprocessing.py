import numpy as np
from PIL import Image
from src.config import IMAGE_SIZE, NORMALIZATION_FACTOR
from src.face_detection import detect_and_crop_face


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess an image for model inference."""

    # 1. Detect and crop the face first using OpenCV YuNet 
    image = detect_and_crop_face(image)

    if image is None:
        raise ValueError("No face detected in the image.")

    # 2. Convert to grayscale and resize
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE)

    # 3. Normalize and format for TensorFlow/Keras
    image_array = np.asarray(image, dtype=np.float32)
    image_array /= NORMALIZATION_FACTOR

    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array
