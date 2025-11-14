# ECG Anomaly Detector – Autoencoder (Demo)

This hero app demonstrates a simple autoencoder-based anomaly detector
for a 1D ECG-like signal, built with PyTorch and Streamlit.

## 💓 What it does

- Generates a synthetic ECG-like signal with occasional spikes, **or**
  loads a user-provided CSV with a numeric `value` column.
- Builds sliding windows over the time series.
- Trains a small fully connected autoencoder on the windowed data.
- Computes reconstruction errors for each window.
- Flags high-error windows as potential anomalies, with:
  - A histogram of reconstruction errors.
  - A slider-based anomaly threshold.
  - A plot of the signal with anomalies highlighted.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

- Runs on Streamlit Cloud (CPU).
- Can be deployed on Hugging Face Spaces using the Streamlit SDK.
