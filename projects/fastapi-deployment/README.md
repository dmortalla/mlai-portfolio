# FastAPI Iris Classifier – ML Model Deployment Demo

This project demonstrates how to deploy a simple machine learning model
(RandomForestClassifier trained on the Iris dataset) as a FastAPI service,
with clean request/response schemas and Docker packaging.

It is designed as a recruiter-friendly ML engineering portfolio project.

## Features

- FastAPI app with:
  - `GET /health` – health check
  - `POST /predict` – predict Iris species from sepal/petal measurements
- Pydantic models for request validation and response typing
- Separate training script (`train_model.py`)
- Dockerfile for containerized deployment

## Project Structure

    fastapi-deployment/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py
    │   └── schemas.py
    ├── train_model.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

## Usage

1. Install dependencies:

    pip install -r requirements.txt

2. Train and save the model:

    python train_model.py

3. Run the FastAPI app:

    uvicorn app.main:app --reload

4. Open the interactive docs in your browser:

    http://127.0.0.1:8000/docs

## Docker

Build and run the Docker image:

    docker build -t iris-fastapi .
    docker run -p 8000:8000 iris-fastapi
