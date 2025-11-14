# Anomaly Detection Ensemble – IsolationForest + LOF

This project demonstrates an anomaly detection ensemble that combines
**IsolationForest** and **LocalOutlierFactor (LOF)** on a simple 2D
synthetic dataset.

## Features

- Generates a dataset with:
  - Normal points from a Gaussian distribution
  - Injected anomalies from a wide uniform range
- Fits:
  - IsolationForest
  - LocalOutlierFactor
- Normalizes their anomaly scores and forms an ensemble score
- Flags top scoring points as anomalies
- Visualizes normal vs anomalous points on a scatter plot

## Usage

1. Install dependencies:

    pip install -r requirements.txt

2. Run the script:

    python anomaly_ensemble.py

A Matplotlib window will appear, showing anomalies highlighted.
