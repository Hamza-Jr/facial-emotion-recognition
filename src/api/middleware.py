import logging
import time
import uuid

from fastapi import Request

logger = logging.getLogger(__name__)


async def logging_middleware(request: Request, call_next):
    """Log HTTP request completion with request ID and duration."""

    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = None

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "http_request_completed",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "status_code": response.status_code if response else 500,
                "duration_ms": round(duration_ms, 2),
            },
        )
