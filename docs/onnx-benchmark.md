# ONNX Benchmark and Inference Decision

## Purpose

This document records the performance comparison between the original Keras
emotion classification model and its ONNX version.

The goal is to determine which inference backend should be used for the
remaining application development.

Both implementations use the same:

- Input image
- Preprocessing pipeline
- Emotion labels
- `EmotionPredictor` interface

The prediction pipeline and model contract remain the same; only the inference
backend differs.

---

## 1. Model Implementations

The application defines a common model abstraction through `ModelBase`.

`EmotionPredictor` depends only on `ModelBase`, which means the prediction
logic is independent of the underlying model implementation.

This allows the application to switch between Keras and ONNX without changing
the prediction logic.

---

## 2. Benchmark Method

The benchmark measures:

- Model loading time
- Prediction time
- Total loading and prediction time

The same test image and preprocessing pipeline are used for both models.

The benchmark was executed in the current development environment.

---

## 3. Benchmark Results

| Metric | Keras | ONNX |
|---|---:|---:|
| Model load time | 502.32 ms | 63.12 ms |
| Prediction time | 606.48 ms | 51.58 ms |
| Total time | 1108.80 ms | 114.70 ms |
| Predicted emotion | Happy | Happy |
| Confidence | 99.99% | 99.99% |

---

## 4. Performance Analysis

### Model Loading

Keras required:

- **502.32 ms**

ONNX required:

- **63.12 ms**

ONNX loaded approximately **8 times faster** in this benchmark.

### Prediction

Keras required:

- **606.48 ms**

ONNX required:

- **51.58 ms**

ONNX inference was approximately **12 times faster** in this benchmark.

### Total Time

Keras required:

- **1108.80 ms**

ONNX required:

- **114.70 ms**

The combined model-loading and prediction measurement was approximately
**9.7 times faster with ONNX**.

---

## 5. Prediction Equivalence

Performance is not the only factor considered.

The Keras and ONNX models were also compared using exactly the same
preprocessed input.

The maximum absolute difference between their output probabilities was:

**0.0000001192**

Both models predicted:

- **Emotion:** Happy
- **Confidence:** 99.99%

The extremely small difference in probabilities indicates that the ONNX
conversion preserves the prediction behavior of the original Keras model for
the tested input.

---

## 6. Decision

### ONNX Runtime

ONNX Runtime will be used as the inference backend for the rest of the project.

The main reasons are:

- Significantly lower model loading time.
- Significantly lower inference time.
- Prediction results are effectively equivalent to the Keras model.
- ONNX Runtime is designed specifically for model inference.
- The existing `ModelBase` abstraction allows ONNX to be used without changing
  `EmotionPredictor`.
- The architecture remains extensible because additional model
  implementations can be added later.

---

## 7. Benchmark Conclusion

The ONNX model preserves the prediction behavior of the original Keras model
while providing significantly better performance in the current development
environment.
