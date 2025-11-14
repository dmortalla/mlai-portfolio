"""Pydantic request/response models for the Iris FastAPI service."""

from pydantic import BaseModel, Field


class IrisObservation(BaseModel):
    """Features for a single Iris flower observation.

    Attributes:
        sepal_length: Sepal length in centimeters.
        sepal_width: Sepal width in centimeters.
        petal_length: Petal length in centimeters.
        petal_width: Petal width in centimeters.
    """

    sepal_length: float = Field(..., gt=0, description="Sepal length in cm (must be > 0).")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm (must be > 0).")
    petal_length: float = Field(..., gt=0, description="Petal length in cm (must be > 0).")
    petal_width: float = Field(..., gt=0, description="Petal width in cm (must be > 0).")


class IrisPredictionResponse(BaseModel):
    """Model prediction for an Iris observation.

    Attributes:
        predicted_class: The predicted Iris species label.
        class_probabilities: Mapping from class label to predicted probability.
    """

    predicted_class: str
    class_probabilities: dict[str, float]
