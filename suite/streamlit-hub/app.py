"""MLAI Streamlit Hub: a central launcher/overview for all Hero apps.

This hub does not try to embed the apps directly. Instead, it:
- Lists each Hero app with a title and description.
- Shows the local command to run the app via Streamlit.
- Leaves GitHub/Hostinger links as fields you can fill in later.

This design keeps things robust and simple for both local use and recruiters.
"""

from dataclasses import dataclass
from typing import List

import streamlit as st


@dataclass
class HeroApp:
    """Metadata for a single Hero app in the MLAI portfolio."""

    name: str
    folder: str
    description: str
    local_command: str
    github_url: str = ""
    hostinger_url: str = ""


def get_hero_apps() -> List[HeroApp]:
    """Return the list of Hero apps configured for the MLAI portfolio.

    You can edit this list to update URLs or descriptions over time.
    """
    return [
        HeroApp(
            name="AI Document Advisor (RAG)",
            folder="rag-document-advisor",
            description=(
                "RAG pipeline that turns uploaded documents into a searchable "
                "AI assistant using embeddings, vector search, and LLM responses."
            ),
            local_command="streamlit run hero/rag-document-advisor/app.py",
            github_url="",  # fill in once the repo is public
            hostinger_url="",  # fill in if deployed
        ),
        HeroApp(
            name="Multimodal RAG Assistant (Image + Text)",
            folder="multimodal-rag-assistant",
            description=(
                "Combines document retrieval with image captioning and LLM "
                "reasoning for multimodal question answering."
            ),
            local_command="streamlit run hero/multimodal-rag-assistant/app.py",
        ),
        HeroApp(
            name="Semantic Search Engine (FAISS)",
            folder="semantic-search-faiss",
            description=(
                "Semantic search over CSV/TXT data using sentence embeddings "
                "and FAISS vector similarity."
            ),
            local_command="streamlit run hero/semantic-search-faiss/app.py",
        ),
        HeroApp(
            name="ALS Recommender (Implicit Feedback)",
            folder="als-recommender",
            description=(
                "Matrix factorization using ALS on implicit feedback to "
                "recommend items to users."
            ),
            local_command="streamlit run hero/als-recommender/app.py",
        ),
        HeroApp(
            name="Time Series Forecaster (LSTM)",
            folder="timeseries-forecaster",
            description=(
                "Univariate time series forecasting with a PyTorch LSTM model "
                "and configurable hyperparameters."
            ),
            local_command="streamlit run hero/timeseries-forecaster/app.py",
        ),
        HeroApp(
            name="Traffic Sign Classifier (CNN Demo)",
            folder="traffic-sign-classifier",
            description=(
                "CNN-based classifier for traffic sign images. Ships in demo "
                "mode until trained weights are added."
            ),
            local_command="streamlit run hero/traffic-sign-classifier/app.py",
        ),
        HeroApp(
            name="ECG Anomaly Detector (Autoencoder)",
            folder="ecg-anomaly-autoencoder",
            description=(
                "Autoencoder-based anomaly detector for ECG-like 1D signals "
                "with reconstruction-error visualization."
            ),
            local_command="streamlit run hero/ecg-anomaly-autoencoder/app.py",
        ),
    ]


def main() -> None:
    """Render the MLAI Streamlit Hub."""
    st.set_page_config(page_title="MLAI Hero Apps Hub", layout="wide")
    st.title("🚀 Machine Learning & AI Engineering – Hero Apps Hub")

    st.markdown(
        """
        This hub showcases your core Machine Learning & AI Engineering hero apps.
        Each tile includes:
        - A short description of what the app does.
        - The local command to run it from the repo root.
        - Optional GitHub / Hostinger links you can fill in later.
        """
    )

    apps = get_hero_apps()

    cols = st.columns(2)
    for i, app in enumerate(apps):
        col = cols[i % 2]
        with col:
            st.markdown(f"### {app.name}")
            st.write(app.description)

            st.code(app.local_command, language="bash")

            if app.github_url:
                st.markdown(f"[View on GitHub]({app.github_url})")
            if app.hostinger_url:
                st.markdown(f"[Live demo on Hostinger]({app.hostinger_url})")

            st.markdown("---")


if __name__ == "__main__":
    main()
