import logging
from io import BytesIO

from fastapi import HTTPException, Request, UploadFile
from PIL import Image

from src.config import (
    MAX_IMAGE_SIZE_BYTES,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_IMAGE_TYPES,
)

logger = logging.getLogger(__name__)


async def validate_image(
    request: Request,
    image: UploadFile,
) -> Image.Image:
    """Validate and decode an uploaded image."""

    request_id = request.state.request_id

    contents = await image.read()

    if not contents:
        logger.warning(
            "invalid_upload",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 400,
            },
        )

        raise HTTPException(
            status_code=400,
            detail="Uploaded image file is empty. Please upload a valid image.",
        )

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        logger.warning(
            "oversized_upload",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 413,
            },
        )

        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Please upload an image no larger than 6 MiB.",
        )

    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        logger.warning(
            "unsupported_media_type",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 415,
            },
        )

        raise HTTPException(
            status_code=415,
            detail="Unsupported image type. Please upload a valid PNG or JPEG image.",
        )

    # Validate file extension
    filename = image.filename or ""
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if file_ext not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.warning(
            "unsupported_file_extension",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 415,
            },
        )

        raise HTTPException(
            status_code=415,
            detail="Unsupported image extension. Please upload a PNG or JPG image.",
        )

    try:
        pil_image = Image.open(BytesIO(contents))
        pil_image.load()

    except (Image.UnidentifiedImageError, OSError) as exc:
        logger.warning(
            "invalid_upload",
            extra={
                "request_id": request_id,
                "endpoint": "/predict",
                "status_code": 400,
            },
        )

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from exc

    return pil_image
