"""HAR sequence classification with an LSTM model (PyTorch).

This script expects a preprocessed Human Activity Recognition dataset in
CSV form and trains a simple LSTM classifier to predict activity labels.

For portability, the script does not ship the dataset. Instead, the user
should provide a CSV file with:
- Feature columns (sensor readings)
- A target column called 'label'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HARData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int


class HARLSTM(nn.Module):
    """Simple LSTM-based classifier for HAR data."""

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def load_har_csv(path: Path, label_col: str = "label") -> HARData:
    """Load HAR data from a CSV file and split into train/test sets.

    Args:
        path: Path to the CSV file.
        label_col: Name of the label column.

    Returns:
        HARData with train/test splits and number of classes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If label column is missing or data is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV.")

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values

    if X.ndim != 2:
        raise ValueError("Expected 2D feature matrix after dropping label.")

    # Example: treat each row as a single timestep sequence of features.
    # For real HAR, you would reconstruct sequences from sliding windows.
    X = X.astype("float32")
    X_seq = X[:, None, :]  # shape: (n_samples, seq_len=1, n_features)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    return HARData(X_train, X_test, y_train, y_test, n_classes=n_classes)


def train_model(
    data: HARData,
    input_size: int,
    hidden_size: int = 64,
    batch_size: int = 32,
    n_epochs: int = 10,
    learning_rate: float = 1e-3,
) -> Tuple[HARLSTM, float]:
    """Train the HARLSTM model on the given data.

    Args:
        data: HARData with train/test splits.
        input_size: Number of features per timestep.
        hidden_size: Hidden dimension of LSTM.
        batch_size: Training batch size.
        n_epochs: Number of epochs.
        learning_rate: Learning rate.

    Returns:
        A tuple of (trained_model, test_accuracy).
    """
    model = HARLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=data.n_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(
        torch.from_numpy(data.X_train),
        torch.from_numpy(data.y_train),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(data.X_test),
        torch.from_numpy(data.y_test),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch + 1}/{n_epochs} - loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    test_acc = correct / max(1, total)
    print(f"Test accuracy: {test_acc:.3f}")
    return model, test_acc


def main() -> None:
    """Entry point for training the HAR LSTM model.

    Expects a CSV file called 'har_data.csv' in the same folder.
    """
    csv_path = Path(__file__).resolve().parent / "har_data.csv"
    data = load_har_csv(csv_path)
    input_size = data.X_train.shape[2]
    _, acc = train_model(data, input_size=input_size)
    print("Training complete.")


if __name__ == "__main__":
    main()
