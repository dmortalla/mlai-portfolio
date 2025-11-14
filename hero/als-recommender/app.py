"""Streamlit ALS recommender demo.

This app demonstrates a simple implicit-feedback recommender system using
matrix factorization (ALS). Users can upload a CSV or use a sample dataset,
train an ALS model, and get top-N recommendations for a selected user.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from recommender import (
    build_interaction_matrix,
    train_als_model,
    get_user_recommendations,
)


def load_sample_data() -> pd.DataFrame:
    """Return a tiny sample implicit-feedback dataset.

    The dataset has the columns:
    - user_id
    - item_id
    - interaction  (e.g., count of views or clicks)

    Returns:
        A pandas DataFrame with sample interactions.
    """
    data = {
        "user_id": [1, 1, 1, 2, 2, 3, 3, 4, 5],
        "item_id": [101, 102, 103, 101, 104, 102, 105, 103, 104],
        "interaction": [3, 1, 2, 4, 1, 2, 5, 1, 3],
    }
    return pd.DataFrame(data)


def main() -> None:
    """Run the Streamlit ALS recommender application.

    This function sets up the UI, loads or accepts uploaded data, trains an
    ALS model on implicit interactions, and displays recommendations.
    """
    st.set_page_config(page_title="ALS Recommender (Implicit)", layout="wide")
    st.title("🎬 ALS Recommender – Matrix Factorization on Implicit Feedback")

    st.markdown(
        "Upload a CSV with implicit feedback (user_id, item_id, interaction), "
        "or use the built-in sample dataset. The app trains an ALS model and "
        "shows top-N recommendations for a chosen user."
    )

    st.sidebar.header("Dataset")
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)

    uploaded_file = None
    if not use_sample:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with columns: user_id, item_id, interaction",
            type=["csv"],
        )

    # Load data
    if use_sample:
        df = load_sample_data()
        st.success("Using built-in sample dataset.")
    else:
        if uploaded_file is None:
            st.warning("Please upload a CSV file or select sample dataset.")
            return
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Error reading CSV file: {exc}")
            return

    st.subheader("Preview of Interaction Data")
    st.dataframe(df.head())

    # Basic column validation
    required_cols = {"user_id", "item_id", "interaction"}
    if not required_cols.issubset(df.columns):
        st.error(
            f"CSV must contain columns: {', '.join(sorted(required_cols))}. "
            f"Found columns: {', '.join(df.columns)}"
        )
        return

    # Cast to appropriate types defensively
    try:
        df["user_id"] = df["user_id"].astype("int64")
        df["item_id"] = df["item_id"].astype("int64")
        df["interaction"] = df["interaction"].astype("float64")
    except Exception as exc:
        st.error(f"Error converting column types: {exc}")
        return

    # ALS hyperparameters
    st.sidebar.header("ALS Hyperparameters")
    factors = st.sidebar.slider("Number of latent factors", 10, 100, 40)
    regularization = st.sidebar.slider("Regularization", 0.001, 1.0, 0.1)
    iterations = st.sidebar.slider("Iterations", 5, 50, 20)
    alpha = st.sidebar.slider("Confidence scaling (alpha)", 1.0, 80.0, 40.0)

    top_n = st.sidebar.slider("Top-N recommendations", 1, 20, 5)

    # Build interaction matrix
    try:
        interaction_matrix, user_mapping, item_mapping = build_interaction_matrix(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    # Train ALS model
    with st.spinner("Training ALS model on implicit feedback..."):
        model = train_als_model(
            interaction_matrix,
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
        )
    st.success("Model trained successfully.")

    # User selection
    available_users = sorted(user_mapping.keys())
    st.subheader("Get Recommendations")
    selected_user_id: Optional[int] = st.selectbox(
        "Select user_id for recommendations:", available_users
    )

    if st.button("Recommend Items"):
        if selected_user_id is None:
            st.warning("Please select a user_id.")
            return

        try:
            recs = get_user_recommendations(
                model,
                interaction_matrix,
                user_id=selected_user_id,
                user_mapping=user_mapping,
                item_mapping=item_mapping,
                top_n=top_n,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.markdown(f"### 🎯 Top {top_n} recommendations for user {selected_user_id}")
        if not recs:
            st.info("No recommendations available for this user.")
            return

        rec_df = pd.DataFrame(recs)
        st.table(rec_df)


if __name__ == "__main__":
    main()
