"""LSTM-based univariate time series utilities for the forecaster app."""

from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    """A simple univariate LSTM regressor for time series forecasting."""

    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1) -> None:
        """Initialize the LSTM regressor.

        Args:
            input_size: Number of input features (1 for univariate series).
            hidden_size: Number of hidden units in the LSTM layer.
            num_layers: Number of stacked LSTM layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        output, _ = self.lstm(x)
        # Use the last time step's output
        last_output = output[:, -1, :]
        out = self.fc(last_output)
        return out.squeeze(-1)


def prepare_univariate_data(
    series: np.ndarray,
    window_size: int = 20,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Prepare sliding-window sequences for univariate time series forecasting.

    Args:
        series: 1D numpy array of time series values.
        window_size: Number of past steps to use as input.
        train_ratio: Fraction of data to use for training.

    Returns:
        A tuple of:
        - X_train: Training input sequences, shape (n_train, window_size, 1)
        - y_train: Training targets, shape (n_train,)
        - X_test: Test input sequences, shape (n_test, window_size, 1)
        - y_test: Test targets, shape (n_test,)
        - train_index_end: Index marking the end of the training portion in the original series.

    Raises:
        ValueError: If the series is too short or train_ratio is invalid.
    """
    if series.ndim != 1:
        raise ValueError("Series must be a 1D numpy array.")
    if not (0.5 <= train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0.5 and 1.0.")

    n = len(series)
    if n <= window_size + 1:
        raise ValueError("Time series is too short for the specified window size.")

    # Build all possible (window, target) pairs
    X = []
    y = []
    for i in range(n - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")

    # Split into train/test
    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    train_size = max(1, min(train_size, n_samples - 1))  # guardrail

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # Reshape for LSTM: (batch, seq, features)
    X_train = X_train.reshape((-1, window_size, 1))
    X_test = X_test.reshape((-1, window_size, 1))

    # train_index_end marks the last training target index in original series
    train_index_end = train_size - 1

    return X_train, y_train, X_test, y_test, train_index_end


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 15,
    learning_rate: float = 0.001,
    hidden_size: int = 64,
    batch_size: int = 32,
) -> Tuple[LSTMRegressor, List[float]]:
    """Train an LSTMRegressor on the given training data.

    Args:
        X_train: Training input, shape (n_train, window_size, 1).
        y_train: Training targets, shape (n_train,).
        n_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        hidden_size: Hidden size for the LSTM.
        batch_size: Mini-batch size.

    Returns:
        The trained LSTMRegressor model and a list of training losses per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRegressor(input_size=1, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses: List[float] = []

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataset)
        train_losses.append(epoch_loss)

    return model, train_losses


def forecast_with_model(model: LSTMRegressor, X_test: np.ndarray) -> np.ndarray:
    """Generate forecasts from an LSTMRegressor on test sequences.

    Args:
        model: A trained LSTMRegressor instance.
        X_test: Test input sequences, shape (n_test, window_size, 1).

    Returns:
        A 1D numpy array of predictions with shape (n_test,).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).to(device)
        outputs = model(inputs)
        preds = outputs.detach().cpu().numpy()
    return preds
