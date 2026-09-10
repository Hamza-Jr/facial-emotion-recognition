from fastapi.testclient import TestClient

from src.api.main import app


def test_predictor_is_reused_between_requests() -> None:
    """Verify the application reuses the same predictor instance."""

    with TestClient(app) as client:
        predictor_before = app.state.predictor

        client.get("/health")

        predictor_after = app.state.predictor

    assert predictor_before is predictor_after


def test_model_initializes_once(monkeypatch) -> None:
    """Verify the model is initialized once during application startup."""

    initialization_count = 0

    original_init = __import__(
        "src.models.onnx_model",
        fromlist=["ONNXEmotionModel"],
    ).ONNXEmotionModel.__init__

    def tracked_init(self, *args, **kwargs):
        nonlocal initialization_count

        initialization_count += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(
        "src.models.onnx_model.ONNXEmotionModel.__init__",
        tracked_init,
    )

    with TestClient(app):
        pass

    assert initialization_count == 1


def test_predict_does_not_initialize_model(monkeypatch) -> None:
    """Verify /predict does not initialize a new model."""

    initialization_count = 0

    original_init = (
        __import__(
            "src.models.onnx_model",
            fromlist=["ONNXEmotionModel"],
        )
        .ONNXEmotionModel.__init__
    )

    def tracked_init(self, *args, **kwargs):
        nonlocal initialization_count

        initialization_count += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(
        "src.models.onnx_model.ONNXEmotionModel.__init__",
        tracked_init,
    )

    with TestClient(app) as client:
        initialization_count = 0

        with open("tests/fixtures/test.png", "rb") as image:
            response = client.post(
                "/predict",
                files={"image": ("test.png", image, "image/png")},
            )

    assert response.status_code == 200
    assert initialization_count == 0
