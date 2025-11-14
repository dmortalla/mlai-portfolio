# Graph Neural Network – Cora Node Classification (GCN, PyTorch Geometric)

This project trains a small **Graph Convolutional Network (GCN)** on the
Cora citation network using **PyTorch Geometric**.

## Features

- Uses the `Planetoid` dataset loader to download Cora.
- Two-layer GCN with dropout and ReLU activation.
- Reports training, validation, and final test accuracy.
- Runs on GPU if available, otherwise CPU.

## Requirements

- `torch`
- `torch-geometric` and its dependencies (install instructions differ by OS).
  See the official PyTorch Geometric website for the correct install command.

## Usage

    pip install torch
    # Then install torch-geometric per instructions from:
    # https://pytorch-geometric.readthedocs.io/

    python gnn_cora.py

The script will download Cora to `./data/Cora` by default and train the GCN.
