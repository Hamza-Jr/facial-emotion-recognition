# Model Architecture
## Purpose

The goal of this architecture is to decouple the prediction logic from the underlying model implementation.

This allows the application to use different model types without changing the prediction logic.

Architecture
              ModelBase
                 │
        ┌────────┴────────┐
        │                 │
   Model A            Model B
        │                 │
        └────────┬────────┘
                 │
          EmotionPredictor



## Responsibilities
### ModelBase

Defines the common interface that every model implementation must follow:

predict(inputs: np.ndarray) -> np.ndarray

### Model Implementation

Responsible for:

Loading the model.
Running inference.
Returning prediction probabilities.

### EmotionPredictor

Responsible for:

Preprocessing the input image.
Calling the model.
Selecting the predicted emotion.
Calculating confidence.

The predictor depends on ModelBase, not on a specific model implementation.

### Dependency Injection

The model is provided to the predictor:

model = SomeModel(model_path)
predictor = EmotionPredictor(model)


This allows EmotionPredictor to work with any implementation that follows ModelBase.

### Benefits
Loose coupling — prediction logic is independent of the model.
Extensibility — new model implementations can be added easily.
Testability — components can be tested independently.
Maintainability — model-specific logic stays isolated.