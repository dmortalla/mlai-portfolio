"""Streamlit app for a simple univariate time series forecaster using an LSTM model.

The app allows the user to:
- Upload a CSV file with a datetime column and a numeric target.
- Or use a small built-in sample time series.
- Configure window size, train/test split, and training epochs.
- Train an LSTM-based regressor and view forecast vs. actual values.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from timeseries_model import (
    prepare_univariate_data,
    LSTMRegressor,
    train_lstm_model,
    forecast_with_model,
)


def load_sample_timeseries() -> pd.DataFrame:
    """Generate a simple synthetic daily time series with trend + seasonality.

    Returns:
        DataFrame with columns: 'date' and 'value'.
    """
    rng = pd.date_range("2020-01-01", periods=200, freq="D")
    # Trend + weekly seasonality + noise
    t = np.arange(len(rng))
    trend = 0.05 * t
    seasonality = 2 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(scale=0.5, size=len(rng))
    values = 10 + trend + seasonality + noise
    return pd.DataFrame({"date": rng, "value": values})


def main() -> None:
    """Run the Streamlit LSTM time series forecaster application."""
    st.set_page_config(page_title="Time Series Forecaster (LSTM)", layout="wide")
    st.title("📈 Time Series Forecaster – LSTM (Univariate)")

    st.markdown(
        "Upload a time series CSV or use the sample data, then train an LSTM "
        "model to forecast future values of a single numeric target."
    )

    st.sidebar.header("Dataset")
    use_sample = st.sidebar.checkbox("Use sample time series", value=True)

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with a datetime column and a numeric target column",
            type=["csv"],
        )

    # Load data
    if use_sample:
        df = load_sample_timeseries()
        st.success("Using built-in synthetic daily time series.")
    else:
        if uploaded_file is None:
            st.warning("Please upload a CSV file or select the sample dataset.")
            return
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Error reading CSV file: {exc}")
            return

    st.subheader("Preview of Time Series Data")
    st.dataframe(df.head())

    if df.empty:
        st.error("The dataset is empty. Please provide valid data.")
        return

    # Select datetime and target columns
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    # If no datetime dtypes, try to parse object columns
    if not datetime_cols:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            # Try parsing the first object column as dates, but allow user choice
            for col in obj_cols:
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    continue
            datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    if not datetime_cols:
        st.error("No datetime-like columns found. Please ensure your CSV has a date column.")
        return

    datetime_col = st.selectbox("Select datetime column:", datetime_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found for forecasting.")
        return

    target_col = st.selectbox("Select numeric target column to forecast:", numeric_cols)

    # Sort by datetime
    df = df.sort_values(datetime_col)
    df = df[[datetime_col, target_col]].dropna()
    df = df.reset_index(drop=True)

    st.line_chart(df.set_index(datetime_col)[target_col])

    st.sidebar.header("Model Settings")
    window_size = st.sidebar.slider("Window size (lookback steps)", min_value=5, max_value=60, value=20)
    train_ratio = st.sidebar.slider("Train set ratio", min_value=0.5, max_value=0.95, value=0.8)
    n_epochs = st.sidebar.slider("Training epochs", min_value=5, max_value=50, value=15)
    hidden_size = st.sidebar.slider("Hidden size", min_value=16, max_value=128, value=64)
    learning_rate = st.sidebar.select_slider(
        "Learning rate",
        options=[0.0005, 0.001, 0.002, 0.005],
        value=0.001,
    )

    if st.button("Train LSTM Forecaster"):
        series = df[target_col].values.astype("float32")

        # Prepare sequences
        try:
            (
                X_train,
                y_train,
                X_test,
                y_test,
                train_index_end,
            ) = prepare_univariate_data(series, window_size=window_size, train_ratio=train_ratio)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.info(
            f"Prepared {len(X_train)} training sequences and {len(X_test)} test sequences "
            f"with window size {window_size}."
        )

        with st.spinner("Training LSTM model..."):
            model, train_losses = train_lstm_model(
                X_train,
                y_train,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
            )

        st.success("Training complete.")

        # Plot training loss
        fig, ax = plt.subplots()
        ax.plot(train_losses, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Loss")
        st.pyplot(fig)

        # Forecast on test set
        preds = forecast_with_model(model, X_test)

        # Build comparison DataFrame
        # y_test is aligned with the last part of the series
        test_dates = df[datetime_col].iloc[train_index_end + window_size : train_index_end + window_size + len(y_test)]
        comparison_df = pd.DataFrame(
            {
                "date": test_dates.values,
                "actual": y_test,
                "predicted": preds,
            }
        )

        st.subheader("Forecast vs Actual (Test Set)")
        st.dataframe(comparison_df.head())

        # Plot actual vs predicted
        fig2, ax2 = plt.subplots()
        ax2.plot(comparison_df["date"], comparison_df["actual"], label="Actual")
        ax2.plot(comparison_df["date"], comparison_df["predicted"], label="Predicted")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Value")
        ax2.set_title("LSTM Forecast vs Actual (Test Set)")
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
