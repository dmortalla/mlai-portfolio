# MLflow Tracking – Iris Experiment Demo

This project demonstrates how to use **MLflow** for experiment tracking
and model logging on the classic Iris dataset.

## Features

- Trains two models:
  - Logistic Regression
  - Random Forest
- Logs:
  - Parameters (e.g., `model_type`, `n_estimators`)
  - Metrics (test accuracy)
  - Serialized model artifacts
- Organizes runs under a single experiment: `iris-mlflow-demo`
- Can be inspected with the `mlflow ui` web interface

## Files

- `train_mlflow.py` – Main script to train and log models
- `README.md` – Project description
- `requirements.txt` – Dependencies

## Usage

1. Install dependencies:

    pip install -r requirements.txt

2. Run the training script:

    python train_mlflow.py

3. Start the MLflow UI:

    mlflow ui

   Then open http://127.0.0.1:5000 in your browser to explore runs.

This project is a compact, recruiter-friendly example of ML experiment
tracking and is a natural precursor to integrating a model registry
with FastAPI or other serving layers.
