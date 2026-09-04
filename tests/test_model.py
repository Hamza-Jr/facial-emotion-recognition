from src.config import KERAS_MODEL_PATH
from src.models.keras_model import KerasEmotionModel


def test_model_loads() -> None:
    """Verify that the configured emotion model loads successfully."""
    KerasEmotionModel(KERAS_MODEL_PATH)
