from pathlib import Path

import numpy as np
from PIL import Image

from src.config import KERAS_MODEL_PATH, ONNX_MODEL_PATH
from src.models.keras_model import KerasEmotionModel
from src.models.onnx_model import ONNXEmotionModel
from src.preprocessing import preprocess_image


INPUT_IMAGE = Path("tests/fixtures/test.png")


def main() -> None:
    """Compare Keras and ONNX prediction probabilities."""

    image = Image.open(INPUT_IMAGE)

    # Preprocess the image once so both models receive
    # exactly the same input.
    processed_image = preprocess_image(image)

    keras_model = KerasEmotionModel(KERAS_MODEL_PATH)
    onnx_model = ONNXEmotionModel(ONNX_MODEL_PATH)

    keras_output = keras_model.predict(processed_image)
    onnx_output = onnx_model.predict(processed_image)

    keras_probabilities = keras_output[0]
    onnx_probabilities = onnx_output[0]

    print("Keras probabilities:")
    print(keras_probabilities)

    print("\nONNX probabilities:")
    print(onnx_probabilities)

    differences = np.abs(
        keras_probabilities - onnx_probabilities
    )

    print("\nAbsolute differences:")
    print(differences)

    print(f"\nMaximum difference: {np.max(differences):.10f}")


if __name__ == "__main__":
    main()
