from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.models.base import ModelBase


class ONNXEmotionModel(ModelBase):
    """ONNX Runtime implementation of the emotion classification model."""

    def __init__(self, model_path: Path | str) -> None:
        self._model_path = Path(model_path)
        self._session = self._load_model()

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _load_model(self) -> ort.InferenceSession:
        """Load the ONNX emotion model."""
        return ort.InferenceSession(
            str(self._model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run emotion inference using the ONNX model."""
        return self._session.run(
            [self._output_name],
            {self._input_name: inputs},
        )[0]
