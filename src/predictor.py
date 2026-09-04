import numpy as np
from PIL import Image

from src.config import EMOTION_LABELS
from src.models.base import ModelBase
from src.preprocessing import preprocess_image


class EmotionPredictor:
    """Predict emotions using an injected model implementation."""

    def __init__(self, model: ModelBase) -> None:
        self._model = model

    def predict_emotion(self, image: Image.Image) -> dict[str, str | float]:
        """Predict the emotion and confidence for an input image."""

        processed_image = preprocess_image(image)

        probabilities = self._model.predict(processed_image)[0]

        predicted_class = int(np.argmax(probabilities))
        emotion = EMOTION_LABELS[predicted_class]
        confidence = float(probabilities[predicted_class])

        return {
            "emotion": emotion,
            "confidence": confidence,
        }
