"""Utility functions for an ALS-based implicit-feedback recommender.

Uses the `implicit` library's AlternatingLeastSquares implementation.
"""

from typing import Dict, Tuple, List, Any

import numpy as np
import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares


def build_interaction_matrix(df):
    """Build a sparse user-item interaction matrix from a DataFrame.

    The DataFrame must have columns:
    - 'user_id'
    - 'item_id'
    - 'interaction'

    Args:
        df: A pandas DataFrame with implicit feedback.

    Returns:
        A tuple of:
        - interaction_matrix (scipy.sparse.csr_matrix)
        - user_mapping (dict: original user_id -> row index)
        - item_mapping (dict: original item_id -> col index)

    Raises:
        ValueError: If the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty; cannot build interaction matrix.")

    # Create mappings
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()

    user_mapping = {uid: i for i, uid in enumerate(unique_users)}
    item_mapping = {iid: j for j, iid in enumerate(unique_items)}

    # Map ids to indices
    user_indices = df["user_id"].map(user_mapping).values
    item_indices = df["item_id"].map(item_mapping).values
    interactions = df["interaction"].astype("float64").values

    # Build sparse matrix
    matrix = sps.csr_matrix(
        (interactions, (user_indices, item_indices)),
        shape=(len(unique_users), len(unique_items)),
    )
    return matrix, user_mapping, item_mapping


def train_als_model(
    interaction_matrix,
    factors: int = 40,
    regularization: float = 0.1,
    iterations: int = 20,
    alpha: float = 40.0,
) -> AlternatingLeastSquares:
    """Train an ALS model on implicit feedback.

    Args:
        interaction_matrix: A scipy CSR sparse matrix of shape (n_users, n_items).
        factors: Number of latent factors.
        regularization: Regularization term for ALS.
        iterations: Number of training iterations.
        alpha: Confidence scaling factor for implicit data.

    Returns:
        A fitted AlternatingLeastSquares model.
    """
    # Convert to the confidence-weighted matrix used by implicit
    # (i.e., C = 1 + alpha * R)
    confidence = (interaction_matrix * alpha).astype("float32")

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
    )

    # implicit expects item-user matrices for fitting
    model.fit(confidence.T)
    return model


def get_user_recommendations(
    model: AlternatingLeastSquares,
    interaction_matrix,
    user_id: int,
    user_mapping: Dict[int, int],
    item_mapping: Dict[int, int],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Get top-N item recommendations for a given user.

    Args:
        model: A trained AlternatingLeastSquares model.
        interaction_matrix: The user-item interaction CSR matrix.
        user_id: Original user_id to recommend for.
        user_mapping: Mapping from original user_id to row index.
        item_mapping: Mapping from original item_id to column index.
        top_n: Number of recommendations to return.

    Returns:
        A list of dictionaries with keys:
        - 'item_id': original item_id
        - 'score': model's confidence score

    Raises:
        ValueError: If the user_id is not found in user_mapping.
    """
    if user_id not in user_mapping:
        raise ValueError(f"user_id {user_id} not found in user_mapping.")

    user_index = user_mapping[user_id]

    # implicit's recommend expects (user, user_items) where user_items is
    # the row of interactions for that user.
    user_items = interaction_matrix[user_index]

    item_ids = list(item_mapping.keys())
    index_to_item = {idx: iid for iid, idx in item_mapping.items()}

    recommended = model.recommend(
        userid=user_index,
        user_items=user_items,
        N=top_n,
        filter_already_liked_items=True,
    )

    results: List[Dict[str, Any]] = []
    for item_index, score in recommended:
        original_item_id = index_to_item.get(item_index)
        results.append(
            {
                "item_id": int(original_item_id) if original_item_id is not None else None,
                "score": float(score),
            }
        )
    return results
