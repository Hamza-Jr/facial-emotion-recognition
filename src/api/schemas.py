from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint."""

    status: str = Field(
        description="Current API health status."
    )


class PredictionResponse(BaseModel):
    """Response schema for the emotion prediction endpoint."""

    emotion: str = Field(
        description="Predicted facial emotion."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Prediction confidence between 0.0 and 1.0."
    )
