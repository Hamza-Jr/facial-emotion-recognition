from io import BytesIO

from fastapi import APIRouter, Depends, File, UploadFile
from PIL import Image

from src.api.dependencies import get_predictor
from src.api.schemas import HealthResponse, PredictionResponse
from src.predictor import EmotionPredictor


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return the health status of the API."""

    return HealthResponse(status="healthy")


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    predictor: EmotionPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict the emotion from an uploaded image."""

    contents = await image.read()
    pil_image = Image.open(BytesIO(contents))

    result = predictor.predict_emotion(pil_image)

    return PredictionResponse(
        emotion=result["emotion"],
        confidence=result["confidence"],
    )