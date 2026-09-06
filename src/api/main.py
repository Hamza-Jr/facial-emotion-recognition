from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="API for facial emotion recognition.",
    version="1.0.0",
)

app.include_router(router)
