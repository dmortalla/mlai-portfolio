# HAR Sequence Model – LSTM Classifier (PyTorch)

This project demonstrates a simple sequence model for Human Activity
Recognition (HAR) using an LSTM-based classifier in PyTorch.

## Dataset Assumption

For portability, the repository does **not** ship the HAR dataset.
Instead, it expects a CSV file named `har_data.csv` in this folder with:

- Multiple numeric feature columns (sensor readings)
- A label column named `label` containing activity names

Example columns:

- `acc_x`, `acc_y`, `acc_z`, `gyro_x`, ... (features)
- `label` (activity string, e.g., WALKING, SITTING)

## Files

- `har_lstm.py` – Model definition, data loader, and training loop.
- `README.md` – Explanation and usage.
- `requirements.txt` – Dependencies.

## Usage

1. Place your preprocessed HAR CSV file as `har_data.csv` here.
2. Install dependencies:

    pip install -r requirements.txt

3. Train the model:

    python har_lstm.py

The script will print training loss per epoch and final test accuracy.
