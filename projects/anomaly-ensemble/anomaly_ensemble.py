"""IsolationForest + LOF anomaly detection ensemble demo.

This script generates a synthetic 2D dataset with normal points and
injected anomalies, fits both IsolationForest and LocalOutlierFactor,
and computes an ensemble anomaly score.

It is built as a simple, readable anomaly detection portfolio project.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class AnomalyResult:
    """Container for anomaly scores and labels."""

    scores_iforest: np.ndarray
    scores_lof: np.ndarray
    scores_ensemble: np.ndarray
    is_anomaly: np.ndarray


def generate_data(n_normal: int = 300, n_anomalies: int = 20, random_state: int = 42):
    """Generate a simple 2D dataset with normal points and anomalies.

    Args:
        n_normal: Number of normal samples.
        n_anomalies: Number of anomaly points.
        random_state: Seed for reproducibility.

    Returns:
        A 2D numpy array X of shape (n_samples, 2).
    """
    rng = np.random.RandomState(random_state)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 2))
    anomalies = rng.uniform(low=-6, high=6, size=(n_anomalies, 2))
    X = np.vstack([normal, anomalies])
    return X


def detect_anomalies(X: np.ndarray, contamination: float = 0.05) -> AnomalyResult:
    """Fit IsolationForest and LOF models and compute ensemble scores.

    Args:
        X: Data array of shape (n_samples, n_features).
        contamination: Proportion of outliers in the data.

    Returns:
        An AnomalyResult with scores and boolean anomaly flags.
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    # IsolationForest gives anomaly scores (higher = more normal in sklearn);
    # we'll invert them so higher means more anomalous.
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    scores_iforest_raw = iso.decision_function(X)
    scores_iforest = -scores_iforest_raw

    # LOF negative_outlier_factor_: smaller = more abnormal; invert sign
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_labels = lof.fit_predict(X)
    scores_lof = -lof.negative_outlier_factor_

    # Normalize scores to [0, 1] for combination
    def min_max_norm(arr: np.ndarray) -> np.ndarray:
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)

    scores_iforest_norm = min_max_norm(scores_iforest)
    scores_lof_norm = min_max_norm(scores_lof)

    scores_ensemble = 0.5 * scores_iforest_norm + 0.5 * scores_lof_norm

    # Threshold at percentile to decide anomalies
    threshold = np.percentile(scores_ensemble, 100 * (1 - contamination))
    is_anomaly = scores_ensemble >= threshold

    return AnomalyResult(
        scores_iforest=scores_iforest_norm,
        scores_lof=scores_lof_norm,
        scores_ensemble=scores_ensemble,
        is_anomaly=is_anomaly,
    )


def plot_results(X: np.ndarray, result: AnomalyResult) -> None:
    """Plot the dataset with anomalies highlighted.

    Args:
        X: Data array of shape (n_samples, 2).
        result: An AnomalyResult with anomaly flags.
    """
    normal_points = X[~result.is_anomaly]
    anomaly_points = X[result.is_anomaly]

    plt.figure(figsize=(6, 6))
    plt.scatter(normal_points[:, 0], normal_points[:, 1], s=15, label="Normal")
    plt.scatter(
        anomaly_points[:, 0], anomaly_points[:, 1],
        s=40, marker="x", label="Anomaly"
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("IsolationForest + LOF Ensemble Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the anomaly detection demo end-to-end."""
    X = generate_data()
    result = detect_anomalies(X)
    print(f"Detected {result.is_anomaly.sum()} anomalies out of {len(X)} samples.")
    plot_results(X, result)


if __name__ == "__main__":
    main()
