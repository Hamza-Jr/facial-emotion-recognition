import pytest

from src.config import KERAS_MODEL_PATH, ONNX_MODEL_PATH
from src.models.keras_model import KerasEmotionModel
from src.models.onnx_model import ONNXEmotionModel


@pytest.mark.parametrize(
    "model_class, model_path",
    [
        (KerasEmotionModel, KERAS_MODEL_PATH),
        (ONNXEmotionModel, ONNX_MODEL_PATH),
    ],
)
def test_model_loads(model_class, model_path) -> None:
    """Verify that each configured emotion model loads successfully."""
    model_class(model_path)
