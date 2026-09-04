from abc import ABC, abstractmethod

import numpy as np


class ModelBase(ABC):
    """Interface for emotion classification models."""

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed inputs."""
        ...
