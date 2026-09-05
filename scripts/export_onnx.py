import numpy as np
import tensorflow as tf

from src.config import KERAS_MODEL_PATH, MODELS_DIR


def export_model() -> None:
    """Export the Keras emotion model to ONNX format."""
    output_path = MODELS_DIR / "emotion_recognition_model.onnx"

    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Keras model not found: {KERAS_MODEL_PATH}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Keras model: {KERAS_MODEL_PATH}")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)

    # Build/call the model using the production input contract.
    dummy_input = np.zeros(
        (1, 48, 48, 1),
        dtype=np.float32,
    )
    model(dummy_input, training=False)

    print(f"Exporting ONNX model to: {output_path}")

    if output_path.exists():
        output_path.unlink()

    model.export(
        str(output_path),
        format="onnx",
    )

    print(f"ONNX model exported successfully: {output_path}")


if __name__ == "__main__":
    export_model()
