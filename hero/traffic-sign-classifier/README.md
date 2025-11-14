# Traffic Sign Classifier – CNN (Demo)

This hero app demonstrates a simple convolutional neural network (CNN)
for classifying traffic sign images, wrapped in a Streamlit interface.

## 🚦 What it does

- Defines a small CNN in PyTorch (see `model.py`).
- Looks for a file named `traffic_sign_cnn.pth` in the app folder:
  - If found, loads those weights as a trained model.
  - If not found, uses randomly initialized weights (demo mode).
- Lets you upload an image of a traffic sign (`.png` / `.jpg` / `.jpeg`).
- Preprocesses the image (resize, normalize) and runs it through the model.
- Shows the top predicted classes with their probabilities.

## ⚠️ Important

Out of the box, this repo does **not** ship a trained model file.
That means predictions will be random unless you:

1. Train the model on a real traffic sign dataset (e.g., GTSRB) offline.
2. Save the trained weights as `traffic_sign_cnn.pth` in this folder.

The code, structure, and guardrails are production-ready and
recruiter-friendly, but you are expected to add your own trained weights.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

- Runs on Streamlit Cloud (CPU).
- Can also be deployed on Hugging Face Spaces using the Streamlit SDK.
