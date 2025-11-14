"""FastAPI service for a simple Iris classifier (Random Forest).

This module exposes a small REST API with the following endpoints:

- GET /health: Basic health check.
- POST /predict: Predict Iris species from sepal/petal measurements.

The API loads a pre-trained model from disk if available. If the model file
is missing, it raises a clear error prompting the user to run the training
script first.
"""

from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import IrisObservation, IrisPredictionResponse


MODEL_PATH = Path(__file__).resolve().parent / "iris_random_forest.joblib"


app = FastAPI(
    title="Iris Classifier API",
    description=(
        "Simple FastAPI service that wraps a RandomForestClassifier trained "
        "on the Iris dataset. Designed for portfolio and deployment demos."
    ),
    version="1.0.0",
)

# Allow all origins by default (for demo purposes).
# Tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Load a pre-trained scikit-learn model from disk.

    Returns:
        The loaded model object.

    Raises:
        HTTPException: If the model file is missing or cannot be loaded.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Model file not found. Please run 'train_model.py' "
                "to train and save the model before calling /predict."
            ),
        )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # guardrail: corrupted or invalid file
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {exc}",
        ) from exc
    return model


@app.get("/health", summary="Health check")
async def health() -> dict:
    """Return a simple health status.

    This endpoint can be used by monitoring tools or load balancers to check
    if the service is up.
    """
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=IrisPredictionResponse,
    summary="Predict Iris species from flower measurements",
)
async def predict(observation: IrisObservation) -> IrisPredictionResponse:
    """Predict the Iris species for a single observation.

    Args:
        observation: An IrisObservation containing sepal and petal measurements.

    Returns:
        An IrisPredictionResponse with the predicted class and probabilities.
    """
    model = load_model()

    # Prepare input as a 2D array for scikit-learn
    features = [
        [
            observation.sepal_length,
            observation.sepal_width,
            observation.petal_length,
            observation.petal_width,
        ]
    ]

    try:
        pred_class_idx = int(model.predict(features)[0])
        probas = model.predict_proba(features)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc

    if not hasattr(model, "classes_"):
        raise HTTPException(
            status_code=500,
            detail="Model is missing 'classes_' attribute. Was it trained correctly?",
        )

    classes: List[str] = [str(c) for c in model.classes_]
    class_probabilities = {
        class_label: float(prob)
        for class_label, prob in zip(classes, probas)
    }

    predicted_label = classes[pred_class_idx]

    return IrisPredictionResponse(
        predicted_class=predicted_label,
        class_probabilities=class_probabilities,
    )
