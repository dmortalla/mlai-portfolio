"""Train a RandomForestClassifier on the Iris dataset and save it to disk.

This script is intentionally simple and self-contained so that recruiters can
read and run it easily. It trains a model using scikit-learn, evaluates it
briefly, and saves it as `app/iris_random_forest.joblib`.
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train_and_save_model() -> None:
    """Train a RandomForestClassifier on the Iris dataset and save the model.

    The model is saved under `app/iris_random_forest.joblib`.
    """
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")

    model_path = Path(__file__).resolve().parent / "app" / "iris_random_forest.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_and_save_model()
