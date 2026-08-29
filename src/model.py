import os

# Suppress TensorFlow startup logs and oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from config import MODEL_PATH


def load_model() -> tf.keras.Model:
    """Load the trained emotion recognition model."""
    return tf.keras.models.load_model(MODEL_PATH)
