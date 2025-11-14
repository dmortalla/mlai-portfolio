"""Streamlit app for ECG-like anomaly detection using an autoencoder.

This app demonstrates:
- Loading or generating a univariate time series (ECG-like signal).
- Training a simple autoencoder on (mostly) normal data.
- Computing reconstruction errors and highlighting potential anomalies.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from autoencoder import (
    build_autoencoder,
    train_autoencoder,
    compute_reconstruction_errors,
    create_sliding_windows,
)


def generate_synthetic_ecg(n_samples: int = 1000) -> pd.DataFrame:
    """Generate a synthetic 1D signal mimicking an ECG-like pattern.

    The signal has a base sine wave plus some noise, with a few injected spikes
    to represent anomalies.

    Args:
        n_samples: Number of time steps.

    Returns:
        DataFrame with columns: 't' (time index), 'value' (signal).
    """
    t = np.arange(n_samples)
    base = np.sin(2 * np.pi * t / 50)  # base rhythm
    noise = np.random.normal(scale=0.1, size=n_samples)
    signal = base + noise

    # Inject a few anomalies as large spikes
    for idx in [200, 500, 800]:
        if idx < n_samples:
            signal[idx] += np.random.normal(loc=3.0, scale=0.5)

    return pd.DataFrame({"t": t, "value": signal})


def main() -> None:
    """Run the ECG anomaly detection app."""
    st.set_page_config(page_title="ECG Anomaly Detector (Autoencoder)", layout="wide")
    st.title("💓 ECG Anomaly Detector – Autoencoder (Demo)")

    st.markdown(
        "This app demonstrates an autoencoder-based anomaly detector for a "
        "1D ECG-like time series. You can upload a CSV with a numeric column, "
        "or use a synthetic example."
    )

    st.sidebar.header("Dataset")
    use_sample = st.sidebar.checkbox("Use synthetic ECG-like signal", value=True)

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with a numeric 'value' column", type=["csv"]
        )

    # Load data
    if use_sample:
        df = generate_synthetic_ecg(n_samples=1000)
        st.success("Using synthetic ECG-like data.")
    else:
        if uploaded_file is None:
            st.warning("Please upload a CSV file or select the synthetic data option.")
            return
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Error reading CSV file: {exc}")
            return

    if df.empty:
        st.error("Dataset is empty. Please provide valid data.")
        return

    # Ensure we have a numeric column 'value'
    if "value" not in df.columns:
        # Try to use the first numeric column as value
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.error("No numeric column found. Ensure your CSV has a 'value' column.")
            return
        df = df.rename(columns={num_cols[0]: "value"})

    st.subheader("Preview of Signal")
    st.dataframe(df.head())

    st.line_chart(df["value"])

    st.sidebar.header("Model Settings")
    window_size = st.sidebar.slider("Window size", min_value=10, max_value=100, value=30)
    n_epochs = st.sidebar.slider("Training epochs", min_value=5, max_value=50, value=15)
    latent_dim = st.sidebar.slider("Latent dimension", min_value=2, max_value=32, value=8)
    learning_rate = st.sidebar.select_slider(
        "Learning rate",
        options=[0.0005, 0.001, 0.002, 0.005],
        value=0.001,
    )

    if st.button("Train Autoencoder and Detect Anomalies"):
        series = df["value"].astype("float32").values

        # Build sliding windows
        try:
            X, centers = create_sliding_windows(series, window_size=window_size)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.info(f"Constructed {len(X)} windows from the series.")

        # Normalize data
        mean = X.mean()
        std = X.std() if X.std() > 0 else 1.0
        X_norm = (X - mean) / std

        with st.spinner("Training autoencoder..."):
            model, train_losses = train_autoencoder(
                X_norm, n_epochs=n_epochs, learning_rate=learning_rate, latent_dim=latent_dim
            )

        st.success("Training complete.")

        # Plot training loss
        fig, ax = plt.subplots()
        ax.plot(train_losses, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Autoencoder Training Loss")
        st.pyplot(fig)

        # Compute reconstruction errors
        errors = compute_reconstruction_errors(model, X_norm)

        st.subheader("Reconstruction Error Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(errors, bins=30)
        ax2.set_xlabel("Reconstruction error")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # Threshold slider
        default_thresh = float(np.percentile(errors, 95))
        threshold = st.slider(
            "Anomaly threshold (higher => fewer anomalies)",
            min_value=float(errors.min()),
            max_value=float(errors.max()),
            value=default_thresh,
        )

        anomaly_flags = errors > threshold
        anomaly_indices = centers[anomaly_flags]

        st.markdown(f"**Detected {anomaly_flags.sum()} anomalies** at window centers.")

        # Plot signal with anomalies highlighted
        fig3, ax3 = plt.subplots()
        ax3.plot(df["value"].values, label="Signal")
        ax3.scatter(
            anomaly_indices,
            df["value"].values[anomaly_indices],
            color="red",
            label="Anomaly",
        )
        ax3.set_xlabel("Time index")
        ax3.set_ylabel("Value")
        ax3.set_title("Signal with Detected Anomalies")
        ax3.legend()
        st.pyplot(fig3)


if __name__ == "__main__":
    main()
