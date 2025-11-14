"""Image preprocessing utilities for the traffic sign classifier app."""

from typing import Tuple

from PIL import Image
import torch
from torchvision import transforms


def get_preprocess_transform(image_size: int = 64) -> transforms.Compose:
    """Return a torchvision transform pipeline for traffic sign images.

    Args:
        image_size: Target square size for resizing the input image.

    Returns:
        A composed torchvision transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image(image: Image.Image, image_size: int = 64) -> torch.Tensor:
    """Preprocess a PIL image into a model-ready tensor.

    Args:
        image: A PIL Image object.
        image_size: Target image size in pixels (square).

    Returns:
        A 4D FloatTensor of shape (1, 3, image_size, image_size).
    """
    transform = get_preprocess_transform(image_size=image_size)
    tensor = transform(image)
    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    return tensor.unsqueeze(0)
