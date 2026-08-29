from unittest.mock import patch

from src.model import load_model


def test_load_model():
    with patch("src.model.tf.keras.models.load_model") as mock_load_model:
        load_model()

        mock_load_model.assert_called_once()
