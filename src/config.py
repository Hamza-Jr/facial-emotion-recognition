from pathlib import Path

# Project root directory 
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

# Model configuration 
KERAS_MODEL_PATH = MODELS_DIR / "emotion_recognition_model.keras"

ONNX_MODEL_PATH = MODELS_DIR / "emotion_recognition_model.onnx"

# YuNet model configuration
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"


# Image configuration
IMAGE_SIZE = (48, 48)
IMAGE_CHANNELS = 1
NORMALIZATION_FACTOR = 255.0

NUM_CLASSES = 7

# Emotion label mapping classes
EMOTION_LABELS = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

