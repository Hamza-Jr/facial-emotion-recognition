from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.dependencies import get_predictor
from src.api.image_validator import validate_image
from src.api.schemas import HealthResponse, PredictionResponse
from src.predictor import EmotionPredictor
from src.preprocessing import NoFaceDetectedError


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

    pil_image = await validate_image(image)

    try:
        result = predictor.predict_emotion(pil_image)

    except NoFaceDetectedError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    return PredictionResponse(
        emotion=result["emotion"],
        confidence=result["confidence"],
    )
