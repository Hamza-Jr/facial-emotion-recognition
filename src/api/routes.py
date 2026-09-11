import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.api.dependencies import get_predictor
from src.api.image_validator import validate_image
from src.api.schemas import HealthResponse, PredictionResponse
from src.config import ONNX_MODEL_VERSION
from src.predictor import EmotionPredictor
from src.preprocessing import NoFaceDetectedError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return the health status of the API."""

    return HealthResponse(status="healthy")


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    image: UploadFile = File(...),
    predictor: EmotionPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Predict the emotion from an uploaded image."""

    request_id = request.state.request_id

    logger.info(
        "prediction_requested",
        extra={
            "request_id": request_id,
            "model_version": ONNX_MODEL_VERSION,
            "endpoint": "/predict",
        },
    )

    pil_image = await validate_image(request, image)

    try:
        result = predictor.predict_emotion(pil_image)

    except NoFaceDetectedError as exc:
        logger.warning(
            "prediction_domain_error",
            extra={
                "request_id": request_id,
                "model_version": ONNX_MODEL_VERSION,
                "endpoint": "/predict",
                "reason": "no_face_detected",
                "status_code": 422,
            },
        )

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error(
            "prediction_failed",
            extra={
                "request_id": request_id,
                "model_version": ONNX_MODEL_VERSION,
                "endpoint": "/predict",
                "error_type": type(exc).__name__,
                "status_code": 500,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc

    logger.info(
        "prediction_successful",
        extra={
            "request_id": request_id,
            "model_version": ONNX_MODEL_VERSION,
            "endpoint": "/predict",
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "status_code": 200,
        },
    )

    return PredictionResponse(
        emotion=result["emotion"],
        confidence=result["confidence"],
    )
