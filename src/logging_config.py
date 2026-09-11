import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for field in (
            "request_id",
            "model_version",
            "endpoint",
            "duration_ms",
            "error_type",
            "status_code",
            "emotion",
            "confidence",
            "reason",
        ):
            value = getattr(record, field, None)

            if value is not None:
                log_entry[field] = value

        return json.dumps(log_entry)


def configure_logging() -> None:
    """Configure application-wide structured JSON logging."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
