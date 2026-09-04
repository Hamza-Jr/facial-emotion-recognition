from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.base import ModelBase


class KerasEmotionModel(ModelBase):
    """Keras implementation of the emotion classification model."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = model_path
        self._model = self._load_model()

    def _load_model(self) -> tf.keras.Model:
        """Load the trained Keras emotion model."""
        return tf.keras.models.load_model(self._model_path)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run emotion inference using the Keras model."""
        return self._model.predict(inputs, verbose=0)
