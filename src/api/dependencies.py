from fastapi import Request

from src.predictor import EmotionPredictor


def get_predictor(request: Request) -> EmotionPredictor:
    """Provide the application-wide emotion predictor."""
    return request.app.state.predictor
