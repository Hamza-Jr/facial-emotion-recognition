# API Contract

## Purpose

This document defines the API endpoints, request and response formats, and general error handling for the facial emotion recognition API.

## Health Check

### Endpoint

GET /health

### Purpose

Check whether the API is running.

### Response

Success:

- HTTP 200
- Returns the API health status.

---

## Emotion Prediction

### Endpoint

POST /predict

### Purpose

Predict the emotion of a face from an uploaded image.

### Request

- Content-Type: multipart/form-data
- Field: image
- Input: image file (.jpg, .jpeg, .png, ...)

### Success Response

- HTTP 200
- Returns a JSON response:
  - `emotion`: string representing the predicted emotion
  - `confidence`: float representing the prediction confidence

### Error Handling

The API handles:

- Missing image — HTTP 422
- Empty upload — HTTP 400
- Invalid image — HTTP 400
- Unsupported MIME type — HTTP 415
- Unsupported file extension — HTTP 415
- Image too large — HTTP 413
- No face detected — HTTP 422
- Prediction errors — HTTP 500
