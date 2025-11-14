"""Autoencoder utilities for ECG-like anomaly detection demo."""

from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleAutoencoder(nn.Module):
    """A small fully connected autoencoder for 1D windowed data."""

    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        """Initialize the autoencoder.

        Args:
            input_dim: Dimensionality of the input window.
            latent_dim: Size of the latent representation.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


def create_sliding_windows(series: np.ndarray, window_size: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows over a 1D series.

    Args:
        series: 1D numpy array of values.
        window_size: Number of consecutive points per window.

    Returns:
        A tuple (X, centers) where:
        - X is a 2D array of shape (n_windows, window_size).
        - centers is a 1D array of indices representing the center of each window.

    Raises:
        ValueError: If the series is too short or not 1D.
    """
    if series.ndim != 1:
        raise ValueError("Series must be a 1D numpy array.")
    if len(series) <= window_size:
        raise ValueError("Series is too short for the given window size.")

    windows = []
    centers = []
    for start in range(len(series) - window_size):
        end = start + window_size
        windows.append(series[start:end])
        centers.append(start + window_size // 2)

    X = np.stack(windows).astype("float32")
    centers_arr = np.array(centers, dtype="int64")
    return X, centers_arr


def train_autoencoder(
    X_train: np.ndarray,
    n_epochs: int = 15,
    learning_rate: float = 0.001,
    latent_dim: int = 8,
    batch_size: int = 64,
) -> Tuple[SimpleAutoencoder, List[float]]:
    """Train a SimpleAutoencoder on the given windowed data.

    Args:
        X_train: Training data windows, shape (n_windows, window_size).
        n_epochs: Number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        latent_dim: Size of the latent representation.
        batch_size: Mini-batch size.

    Returns:
        The trained autoencoder model and a list of training losses per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = SimpleAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    losses: List[float] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= len(dataset)
        losses.append(epoch_loss)

    return model, losses


def compute_reconstruction_errors(model: SimpleAutoencoder, X: np.ndarray) -> np.ndarray:
    """Compute reconstruction errors for each window using a trained autoencoder.

    Args:
        model: Trained SimpleAutoencoder.
        X: Input windows, shape (n_windows, window_size).

    Returns:
        A 1D numpy array of reconstruction errors (MSE per window).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X).to(device)
        recons = model(inputs)
        errors = torch.mean((recons - inputs) ** 2, dim=1)
        return errors.detach().cpu().numpy()
