from pathlib import Path

import pytest
from PIL import Image

from src.config import EMOTION_LABELS, KERAS_MODEL_PATH, ONNX_MODEL_PATH
from src.models.keras_model import KerasEmotionModel
from src.models.onnx_model import ONNXEmotionModel
from src.predictor import EmotionPredictor


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "model_class, model_path",
    [
        (KerasEmotionModel, KERAS_MODEL_PATH),
        (ONNXEmotionModel, ONNX_MODEL_PATH),
    ],
)
def test_predict_emotion(model_class, model_path) -> None:
    """Verify that EmotionPredictor works with each model implementation."""
    model = model_class(model_path)
    predictor = EmotionPredictor(model)

    image = Image.open(FIXTURE_DIR / "test.png")

    result = predictor.predict_emotion(image)

    assert isinstance(result, dict)

    assert "emotion" in result
    assert isinstance(result["emotion"], str)
    assert result["emotion"] in EMOTION_LABELS.values()

    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
