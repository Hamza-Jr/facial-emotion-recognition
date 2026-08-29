import numpy as np
from PIL import Image

from src.config import EMOTION_LABELS
from src.model import load_model
from src.preprocessing import preprocess_image


# Load the model once when this module is imported.
model = load_model()


def predict_emotion(image: Image.Image) -> dict[str, str | float]:
    """Predict the emotion and confidence for an input image."""

    processed_image = preprocess_image(image)

    probabilities = model.predict(processed_image, verbose=0)[0]

    predicted_class = int(np.argmax(probabilities))
    emotion = EMOTION_LABELS[predicted_class]
    confidence = float(probabilities[predicted_class])

    return {
        "emotion": emotion,
        "confidence": confidence,
    }
