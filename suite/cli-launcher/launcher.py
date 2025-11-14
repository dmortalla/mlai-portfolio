"""Command-line launcher for MLAI Hero Streamlit apps.

This script expects to be run from the root of your `mlai-portfolio` repo.
It presents a simple text menu of hero apps and, upon selection, invokes
`streamlit run` on the corresponding app entry point.

Example:
    python suite/cli-launcher/launcher.py
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List


@dataclass
class HeroAppCLI:
    """Metadata for a hero app used by the CLI launcher."""

    name: str
    path: str  # relative path to the Streamlit app entry point


def get_hero_apps() -> List[HeroAppCLI]:
    """Return the hero app list for CLI launching.

    Adjust this list if you rename folders or add new apps.
    """
    return [
        HeroAppCLI(
            name="AI Document Advisor (RAG)",
            path="hero/rag-document-advisor/app.py",
        ),
        HeroAppCLI(
            name="Multimodal RAG Assistant (Image + Text)",
            path="hero/multimodal-rag-assistant/app.py",
        ),
        HeroAppCLI(
            name="Semantic Search Engine (FAISS)",
            path="hero/semantic-search-faiss/app.py",
        ),
        HeroAppCLI(
            name="ALS Recommender (Implicit Feedback)",
            path="hero/als-recommender/app.py",
        ),
        HeroAppCLI(
            name="Time Series Forecaster (LSTM)",
            path="hero/timeseries-forecaster/app.py",
        ),
        HeroAppCLI(
            name="Traffic Sign Classifier (CNN Demo)",
            path="hero/traffic-sign-classifier/app.py",
        ),
        HeroAppCLI(
            name="ECG Anomaly Detector (Autoencoder)",
            path="hero/ecg-anomaly-autoencoder/app.py",
        ),
    ]


def print_menu(apps: List[HeroAppCLI]) -> None:
    """Print a numbered menu of hero apps."""
    print("\n=== MLAI Hero Apps Launcher ===")
    for idx, app in enumerate(apps, start=1):
        print(f"{idx}. {app.name}")
    print("0. Exit")


def main() -> None:
    """Run the CLI menu and dispatch to the selected hero app."""
    apps = get_hero_apps()

    while True:
        print_menu(apps)
        choice_str = input("Select an app to launch (0 to exit): ").strip()

        if not choice_str.isdigit():
            print("Please enter a valid number.\n")
            continue

        choice = int(choice_str)
        if choice == 0:
            print("Exiting launcher.")
            break

        if not (1 <= choice <= len(apps)):
            print("Invalid choice. Please try again.\n")
            continue

        selected_app = apps[choice - 1]
        app_path = selected_app.path

        if not os.path.exists(app_path):
            print(f"Error: {app_path} does not exist. Have you uploaded the hero apps?\n")
            continue

        print(f"Launching: {selected_app.name}")
        print(f"Running: streamlit run {app_path}\n")

        try:
            subprocess.run(["streamlit", "run", app_path], check=False)
        except FileNotFoundError:
            print(
                "Error: 'streamlit' command not found. "
                "Make sure Streamlit is installed and on your PATH.\n"
            )


if __name__ == "__main__":
    main()
