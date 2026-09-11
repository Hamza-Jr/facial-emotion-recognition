# Logging Contract

## Purpose

The API uses structured JSON logs for debugging, monitoring, and measuring prediction performance.

## Common Fields

Each log entry should contain:

- `timestamp`
- `level`
- `logger`
- `event`

When applicable:

- `request_id`
- `model_version`
- `endpoint`
- `duration_ms`
- `error_type`
- `status_code`
- `emotion`
- `confidence`
- `reason`

## Log Levels

- `INFO` - Normal application events
- `WARNING` - Expected or invalid conditions
- `ERROR` - Unexpected application failures

## Events

### Application

- `application_started` - API started
- `model_loaded` - ONNX model loaded
- `application_shutdown` - API shutting down

### HTTP

- `http_request_completed` - HTTP request finished

### Prediction

- `prediction_requested` - Prediction started
- `prediction_successful` - Prediction completed successfully
- `prediction_failed` - Unexpected prediction failure

### Validation

- `invalid_upload` - Invalid image or upload
- `unsupported_media_type` - Unsupported MIME type
- `unsupported_file_extension` - Unsupported file extension
- `oversized_upload` - Image exceeds maximum size

### Domain

- `prediction_domain_error` - Expected prediction problem, such as no face detected

## Security

Never log:

- Uploaded image data
- Image bytes
- Passwords
- API keys
- Tokens
- Secrets
- Sensitive request contents

## Log Format

Each log entry must be a single JSON object.

Example:

{
  "timestamp": "2026-09-11T16:30:15.123Z",
  "level": "INFO",
  "logger": "src.api.routes",
  "event": "prediction_successful",
  "request_id": "8f4c7c3a-4f6d-4b0c-9a2e-123456789abc",
  "endpoint": "/predict",
  "emotion": "Happy",
  "confidence": 0.999879
}
