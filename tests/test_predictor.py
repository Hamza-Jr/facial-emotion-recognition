from pathlib import Path

from PIL import Image

from src.config import EMOTION_LABELS
from src.predictor import predict_emotion


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_predict_emotion():
    image = Image.open(FIXTURE_DIR / "test.png")

    result = predict_emotion(image)

    assert isinstance(result, dict)

    assert "emotion" in result
    assert isinstance(result["emotion"], str)
    assert result["emotion"] in EMOTION_LABELS.values()

    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
