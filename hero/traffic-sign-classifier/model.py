"""CNN model definition and loading utilities for the traffic sign classifier app."""

from typing import Tuple

import os

import torch
from torch import nn


# A small list of placeholder traffic sign class names.
# For a real project, replace this with the full list from your dataset.
CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Yield",
    "Stop",
    "No entry",
    "Priority road",
    "Turn right ahead",
    "Turn left ahead",
]


class SimpleTrafficSignCNN(nn.Module):
    """A small convolutional neural network for traffic sign classification."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the CNN.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the CNN.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_cnn_model() -> Tuple[nn.Module, bool]:
    """Create the CNN model and load weights if available.

    The function looks for a file called `traffic_sign_cnn.pth` in the current
    directory. If present, it loads those weights. Otherwise, the model uses
    randomly initialized weights.

    Returns:
        A tuple of (model, has_trained_weights) where:
        - model is an instance of SimpleTrafficSignCNN on CPU.
        - has_trained_weights is True if weights were successfully loaded.
    """
    num_classes = len(CLASS_NAMES)
    model = SimpleTrafficSignCNN(num_classes=num_classes)

    weights_path = "traffic_sign_cnn.pth"
    has_trained_weights = False

    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            has_trained_weights = True
        except Exception:
            # Guardrail: if loading fails, just keep random weights
            has_trained_weights = False

    model.eval()
    return model, has_trained_weights
