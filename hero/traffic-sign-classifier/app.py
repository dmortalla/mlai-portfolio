"""Streamlit app for a simple traffic sign classifier using a CNN model.

This app is designed as a portfolio-ready demo. It:
- Loads a CNN model definition from model.py.
- Tries to load pre-trained weights from `traffic_sign_cnn.pth` if present.
- Accepts an uploaded traffic sign image.
- Applies basic preprocessing and runs a forward pass to produce class scores.

Note:
    Out of the box, if no trained weights file is present, the model will use
    randomly initialized weights and the predictions will not be meaningful.
    For real use, train the model on a traffic sign dataset (e.g., GTSRB) and
    save weights to `traffic_sign_cnn.pth`.
"""

from typing import Optional

import streamlit as st
from PIL import Image

import torch

from model import load_cnn_model, CLASS_NAMES
from preprocess import preprocess_image


def main() -> None:
    """Run the Streamlit traffic sign classifier application."""
    st.set_page_config(page_title="Traffic Sign Classifier", layout="wide")
    st.title("🚦 Traffic Sign Classifier – CNN (Demo)")

    st.markdown(
        "Upload an image of a traffic sign. The app will run it through a CNN "
        "model and display the top predicted classes.\n\n"
        "**Important:** This is a demo app. For real predictions, you must "
        "train the model on a labeled traffic-sign dataset and save the "
        "weights as `traffic_sign_cnn.pth` in this folder."
    )

    # Load model (with or without weights)
    with st.spinner("Loading CNN model..."):
        model, has_trained_weights = load_cnn_model()

    if has_trained_weights:
        st.success("Loaded trained weights from `traffic_sign_cnn.pth`.")
    else:
        st.warning(
            "No trained weights file found. The model is using random weights, "
            "so predictions are for demonstration only."
        )

    uploaded_file = st.file_uploader(
        "Upload a traffic sign image (PNG or JPG)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as exc:
            st.error(f"Error loading image: {exc}")
            return

        st.subheader("Uploaded Image")
        st.image(image, use_column_width=False, width=256)

        if st.button("Classify Sign"):
            with st.spinner("Running inference..."):
                input_tensor = preprocess_image(image)  # shape: (1, 3, H, W)
                model.eval()
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

            # Top-k predictions
            top_k = min(3, len(CLASS_NAMES))
            top_indices = probs.argsort()[::-1][:top_k]

            st.subheader("Predictions")
            for rank, idx in enumerate(top_indices, start=1):
                class_name = CLASS_NAMES[idx]
                score = probs[idx]
                st.write(f"**{rank}. {class_name}** – probability: {score:.3f}")


if __name__ == "__main__":
    main()
