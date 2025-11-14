"""MLflow experiment tracking demo using the Iris dataset.

This script trains two simple models (LogisticRegression and RandomForest)
on the Iris dataset and logs parameters, metrics, and the trained model
artifacts to a local MLflow tracking server.

It is designed as a portfolio-friendly example of ML experiment tracking
and model registry usage.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data():
    """Load Iris dataset and split into train/test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_logistic_regression(experiment_name: str = "iris-mlflow-demo") -> None:
    """Train and log a LogisticRegression model with MLflow.

    Args:
        experiment_name: Name of the MLflow experiment.
    """
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="logistic-regression"):
        model = LogisticRegression(max_iter=200, random_state=42)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 200)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        # Log the model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Logged LogisticRegression run with accuracy={acc:.3f}")


def train_random_forest(experiment_name: str = "iris-mlflow-demo") -> None:
    """Train and log a RandomForestClassifier model with MLflow.

    Args:
        experiment_name: Name of the MLflow experiment.
    """
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random-forest"):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
        )
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Logged RandomForest run with accuracy={acc:.3f}")


def main() -> None:
    """Entry point: train and log two models with MLflow."""
    print("Starting MLflow Iris demo...")
    train_logistic_regression()
    train_random_forest()
    print(
        "Done. You can now run 'mlflow ui' in this folder to view "
        "experiments at http://127.0.0.1:5000"
    )


if __name__ == "__main__":
    main()
