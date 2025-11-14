"""Time series forecasting with a simple Transformer model (PyTorch).

This script creates a synthetic univariate time series and trains a small
Transformer-based regressor to predict the next value from a window of
past values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TimeSeriesTransformer(nn.Module):
    """Simple Transformer encoder for sequence-to-one regression."""

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        h = self.transformer(x)
        last = h[:, -1, :]
        out = self.fc_out(last)
        return out.squeeze(-1)


def generate_series(n_steps: int = 400) -> np.ndarray:
    """Generate a synthetic time series with trend + seasonality."""
    t = np.arange(n_steps)
    trend = 0.01 * t
    season = 0.5 * np.sin(2 * np.pi * t / 20)
    noise = np.random.normal(scale=0.1, size=n_steps)
    return (trend + season + noise).astype("float32")


def create_windows(series: np.ndarray, window_size: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows for sequence-to-one forecasting."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    X = X[..., None]  # (n_samples, window_size, 1)
    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    batch_size: int = 32,
    n_epochs: int = 15,
    lr: float = 1e-3,
) -> Tuple[TimeSeriesTransformer, float]:
    """Train the Transformer model on the time series windows."""
    model = TimeSeriesTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{n_epochs} - train_loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    total_mse = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb)
            mse = torch.mean((preds - yb) ** 2).item()
            total_mse += mse * xb.size(0)
            total += xb.size(0)

    test_mse = total_mse / max(1, total)
    print(f"Test MSE: {test_mse:.4f}")
    return model, test_mse


def main() -> None:
    """Run the time series Transformer demo."""
    series = generate_series(400)
    window_size = 24
    X, y = create_windows(series, window_size=window_size)

    # Train/test split
    n = len(X)
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    _, test_mse = train_model(X_train, y_train, X_test, y_test)
    print("Training complete.")


if __name__ == "__main__":
    main()
