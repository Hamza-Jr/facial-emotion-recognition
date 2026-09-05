import time
from pathlib import Path

from PIL import Image

from src.config import KERAS_MODEL_PATH, ONNX_MODEL_PATH
from src.models.keras_model import KerasEmotionModel
from src.models.onnx_model import ONNXEmotionModel
from src.predictor import EmotionPredictor


INPUT_IMAGE = Path("tests/fixtures/test.png")


def benchmark_model(model_name: str, model) -> None:
    """Benchmark model loading and prediction."""
    print(f"\n{'=' * 50}")
    print(f"{model_name}")
    print(f"{'=' * 50}")

    # Measure model loading time.
    load_start = time.perf_counter()

    model_instance = model()

    load_time = time.perf_counter() - load_start

    # Create predictor after the model has loaded.
    predictor = EmotionPredictor(model_instance)

    image = Image.open(INPUT_IMAGE)

    # Measure prediction time.
    prediction_start = time.perf_counter()

    result = predictor.predict_emotion(image)

    prediction_time = time.perf_counter() - prediction_start

    total_time = load_time + prediction_time

    print(f"Load time:       {load_time * 1000:.2f} ms")
    print(f"Prediction time: {prediction_time * 1000:.2f} ms")
    print(f"Total time:      {total_time * 1000:.2f} ms")
    print(f"Emotion:         {result['emotion']}")
    print(f"Confidence:      {result['confidence']:.2%}")


def main() -> None:
    """Compare Keras and ONNX model performance."""

    benchmark_model(
        "Keras",
        lambda: KerasEmotionModel(KERAS_MODEL_PATH),
    )

    benchmark_model(
        "ONNX",
        lambda: ONNXEmotionModel(ONNX_MODEL_PATH),
    )


if __name__ == "__main__":
    main()
