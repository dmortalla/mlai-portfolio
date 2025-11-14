# Time Series Forecaster – LSTM (Univariate)

This hero app demonstrates a simple univariate time series forecasting workflow
using an LSTM model in PyTorch and a Streamlit UI.

Users can:
- Upload a CSV containing a datetime column and a numeric target column, or
  use a built-in synthetic daily time series.
- Configure the lookback window size, train/test split, LSTM hidden size,
  learning rate, and number of epochs.
- Train an LSTM-based regressor and view forecast vs. actual values on the
  held-out test set.

## 🚀 Features

- Sliding-window sequence generation for univariate series.
- PyTorch LSTM model with configurable hyperparameters.
- Training loop with MSE loss and Adam optimizer.
- Training loss curve visualization.
- Forecast vs actual comparison with line plots.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

- Runs on Streamlit Cloud (CPU).
- Can be deployed on Hugging Face Spaces using Streamlit SDK.
