import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.middleware import logging_middleware
from src.api.routes import router
from src.config import ONNX_MODEL_PATH, ONNX_MODEL_VERSION
from src.logging_config import configure_logging
from src.models.onnx_model import ONNXEmotionModel
from src.predictor import EmotionPredictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    configure_logging()

    logger.info("application_started")

    model = ONNXEmotionModel(ONNX_MODEL_PATH)
    app.state.predictor = EmotionPredictor(model)

    logger.info(
        "model_loaded",
        extra={
            "model_version": ONNX_MODEL_VERSION,
        },
    )

    yield

    logger.info("application_shutdown")
    app.state.predictor = None


app = FastAPI(
    title="Facial Emotion Recognition API",
    description="API for facial emotion recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

app.middleware("http")(logging_middleware)

app.include_router(router)
