"""Graph Neural Network for node classification on the Cora dataset.

Uses PyTorch Geometric's Planetoid loader to fetch the Cora citation network
and trains a small GCN model for node classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def main(dataset_root: str = "./data/Cora") -> None:
    """Train a GCN on the Cora dataset and report accuracy.

    Args:
        dataset_root: Directory where the Cora dataset will be stored.
    """
    dataset = Planetoid(root=dataset_root, name="Cora")
    data = dataset[0].to(DEVICE)

    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=16,
        out_channels=dataset.num_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            preds = logits.argmax(dim=1)

            train_correct = (preds[data.train_mask] == data.y[data.train_mask]).sum()
            train_acc = train_correct.float() / data.train_mask.sum()

            val_correct = (preds[data.val_mask] == data.y[data.val_mask]).sum()
            val_acc = val_correct.float() / data.val_mask.sum()

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"train_acc={train_acc:.3f} | "
                f"val_acc={val_acc:.3f}"
            )

    # Final test accuracy
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        test_correct = (preds[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = test_correct.float() / data.test_mask.sum()
    print(f"Final test accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
