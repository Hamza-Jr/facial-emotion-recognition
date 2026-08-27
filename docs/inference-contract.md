# Inference Contract

## Purpose

This document defines the contract between the production application and
the trained facial emotion recognition model.

It specifies:

- the model artifact and architecture
- the model input requirements
- the image preprocessing pipeline
- the output format
- the emotion class ordering
- the prediction logic
- the external inference/API contract

The goal is to ensure that production inference reproduces the preprocessing
and prediction behavior established during model development.

---
## 1. Model Contract

The model contract defines the internal interface between the preprocessing
pipeline and the trained CNN model.

### Model
- Final production model: `emotion_recognition_model.keras`
- Alternative/legacy artifact: `emotion_recognition_model.h5`
- Model type: CNN
- Framework: TensorFlow / Keras
- Number of output classes: 7
- Final activation: Softmax

### Input
The model expects image tensors with the following properties:

- **Data type:** `float32`
- **Image format:** Grayscale
- **Image dimensions:** `48 × 48` pixels
- **Channels:** 1 channel (Grayscale)
- **Pixel values normalized to:** `[0, 1]`
- **Unbatched tensor shape:** `(48, 48, 1)`
- **Batched tensor shape:** `(1, 48, 48, 1)` (batch_size, height, width, channels).

### Image Preprocessing Contract
- Convert image to grayscale
- Resize to: 48x48 pixels
- Convert pixel values to numeric array (`float32`)
- Pixel normalization: Division by `255.0` (range `0.0` to `1.0`)
- Add batch dimension to ensure shape `(1, 48, 48, 1)`

### Output
- **Number of classes:** 7
- **Class labels:** Angry, Disgust, Fear, Happiness, Sad, Surprise, Neutral
- Class ordering (Matching training notebook code): 
  ```python
  {
      0: 'Angry',
      1: 'Disgust',
      2: 'Fear',
      3: 'Happiness',
      4: 'Sad',
      5: 'Surprise',
      6: 'Neutral'
  }
- **Output shape:** `(1, 7)` (Softmax probabilities)
- **Prediction selection:** `np.argmax()` for highest probability index, mapping to label dict, returning JSON with emotion, confidence, and full probabilities.

## 2. External Inference Contract
- **Input:** Expected image types .jpg .jpeg .png
- **Image Preprocessing:** apply all steps
- **ouput:** {"emotion": "Happy","confidence": 0.96 }


## 3. Phase 5 Implementation Plan
- **preprocessing.py:** Load image, convert to grayscale, resize to 48x48, normalize by dividing by 255.0, and reshape to `(1, 48, 48, 1)`.
- **model.py (or model loader):** Handle loading and caching of the `.keras` model safely.
- **predictor.py:** Orchestrate the end-to-end pipeline (receives image -> calls preprocessing -> runs model -> applies `np.argmax` -> returns structured JSON result).