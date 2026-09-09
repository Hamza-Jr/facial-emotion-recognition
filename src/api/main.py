from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.config import ONNX_MODEL_PATH
from src.models.onnx_model import ONNXEmotionModel
from src.predictor import EmotionPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    print("Loading ONNX model...")
    model = ONNXEmotionModel(ONNX_MODEL_PATH)

    print("Creating EmotionPredictor...")
    app.state.predictor = EmotionPredictor(model)

    print("Predictor ready.")


    yield

    print("Shutting down application...")
    app.state.predictor = None


app = FastAPI(
    title="Facial Emotion Recognition API",
    description="API for facial emotion recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
