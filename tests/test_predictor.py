from pathlib import Path

from PIL import Image

from src.config import EMOTION_LABELS, KERAS_MODEL_PATH
from src.models.keras_model import KerasEmotionModel
from src.predictor import EmotionPredictor


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_predict_emotion() -> None:
    """Verify that EmotionPredictor returns emotion and confidence."""
    model = KerasEmotionModel(KERAS_MODEL_PATH)
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
