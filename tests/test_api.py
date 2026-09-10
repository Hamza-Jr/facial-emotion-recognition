from fastapi.testclient import TestClient

from src.api.main import app


def test_health() -> None:
    """Verify the health endpoint returns a healthy status."""

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_valid_image() -> None:
    """Verify prediction succeeds for a valid image."""

    with TestClient(app) as client:
        with open("tests/fixtures/test.png", "rb") as image:
            response = client.post(
                "/predict",
                files={"image": ("test.png", image, "image/png")},
            )

    assert response.status_code == 200

    data = response.json()
    assert "emotion" in data
    assert "confidence" in data
    assert isinstance(data["emotion"], str)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_missing_image() -> None:
    """Verify missing image returns a validation error."""

    with TestClient(app) as client:
        response = client.post("/predict")

    assert response.status_code == 422


def test_predict_empty_upload() -> None:
    """Verify an empty upload returns a bad request error."""

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "image": (
                    "empty.png",
                    b"",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400



def test_predict_invalid_image() -> None:
    """Verify invalid image data returns a bad request error."""

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "image": (
                    "invalid.png",
                    b"this is not a real image",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400


def test_predict_unsupported_mime_type() -> None:
    """Verify unsupported MIME type returns an unsupported media type error."""

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "image": (
                    "test.txt",
                    b"some text",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415



def test_predict_unsupported_file_extension() -> None:
    """Verify unsupported file extension returns an unsupported media type error."""

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "image": (
                    "test.gif",
                    b"not a real image",
                    "image/png",
                )
            },
        )

    assert response.status_code == 415


def test_predict_oversized_image() -> None:
    """Verify oversized image returns a payload too large error."""

    oversized_data = b"x" * (6 * 1024 * 1024 + 1)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={
                "image": (
                    "large.png",
                    oversized_data,
                    "image/png",
                )
            },
        )

    assert response.status_code == 413


def test_predict_no_face() -> None:
    """Verify an image with no detectable face returns a validation error."""

    with TestClient(app) as client:
        with open("tests/fixtures/no_face.png", "rb") as image:
            response = client.post(
                "/predict",
                files={"image": ("no_face.png", image, "image/png")},
            )

    assert response.status_code == 422


def test_predict_inference_failure(monkeypatch) -> None:
    """Verify prediction failures return an internal server error."""

    def failing_predictor(self, image) -> None:
        raise RuntimeError("Internal model failure")


    monkeypatch.setattr(
        "src.predictor.EmotionPredictor.predict_emotion",
        failing_predictor,
    )

    with TestClient(app) as client:
        with open("tests/fixtures/test.png", "rb") as image:
            response = client.post(
                "/predict",
                files={"image": ("test.png", image, "image/png")},
            )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed."}
