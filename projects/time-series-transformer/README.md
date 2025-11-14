# Time Series Transformer – Univariate Forecasting Demo

This project demonstrates a small **Transformer-based** model for
univariate time series forecasting in PyTorch.

It uses a synthetic series (trend + seasonality + noise) and trains the
model to predict the next value given a window of past values.

## Files

- `ts_transformer.py` – Model, data generation, training loop.
- `requirements.txt` – Dependencies.

## Usage

    pip install -r requirements.txt
    python ts_transformer.py

The script will print training loss and final test MSE.
