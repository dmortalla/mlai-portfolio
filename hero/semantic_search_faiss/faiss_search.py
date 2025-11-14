"""Helper functions for building a FAISS index and running semantic search.

Uses sentence-transformers (MiniLM) to embed text and FAISS for nearest-neighbor search.
"""

from typing import List, Tuple, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Load the embedding model once at module import
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")


def build_index_from_texts(texts: List[str]) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """Build a FAISS index from a list of text strings.

    Args:
        texts: List of text entries to index.

    Returns:
        A tuple of:
        - FAISS index containing embeddings.
        - metadata list, where each item has keys: "id" and "text".

    Raises:
        ValueError: If no texts are provided.
    """
    if not texts:
        raise ValueError("No texts provided to build the index.")

    embeddings = EMBEDDER.encode(texts)
    embeddings = np.array(embeddings).astype("float32")  # guardrail: correct dtype

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    metadata = [{"id": i, "text": txt} for i, txt in enumerate(texts)]
    return index, metadata


def search_similar_texts(
    query: str,
    index: faiss.IndexFlatL2,
    metadata: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Search for the most similar texts to a query using the FAISS index.

    Args:
        query: The search query as a string.
        index: A FAISS index built over the text embeddings.
        metadata: List of metadata dictionaries corresponding to each indexed text.
        top_k: Number of closest matches to return.

    Returns:
        A list of dictionaries, each containing:
        - "text": the matching text.
        - "score": similarity score (lower means closer in L2 distance).

    Raises:
        ValueError: If query is empty or the index/metadata are not initialized.
    """
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if index is None or not metadata:
        raise ValueError("Index or metadata is empty. Build the index first.")

    query_embedding = EMBEDDER.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)
    distances = distances[0]
    indices = indices[0]

    results: List[Dict[str, Any]] = []
    for dist, idx in zip(distances, indices):
        if idx < 0 or idx >= len(metadata):
            continue
        text = metadata[idx]["text"]
        results.append({"text": text, "score": float(dist)})

    return results
