"""Lightweight RAG (Retrieval-Augmented Generation) utilities.

For simplicity and portability, this implementation uses a TF-IDF vectorizer
stored on disk rather than deep embeddings. It supports:

- Indexing uploaded text / markdown / PDF files.
- Querying the indexed chunks to retrieve top-matching snippets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import PyPDF2  # type: ignore
except Exception:  # pragma: no cover - optional
    PyPDF2 = None  # type: ignore


INDEX_FILENAME = "tfidf_index.pkl"


@dataclass
class RagIndex:
    """Container for a TF-IDF index and its metadata."""

    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    chunks: List[str]
    sources: List[str]


def _load_index(docs_dir: Path) -> Optional[RagIndex]:
    index_path = docs_dir / INDEX_FILENAME
    if not index_path.exists():
        return None
    with index_path.open("rb") as f:
        data = pickle.load(f)
    return RagIndex(
        vectorizer=data["vectorizer"],
        matrix=data["matrix"],
        chunks=data["chunks"],
        sources=data["sources"],
    )


def _save_index(docs_dir: Path, index: RagIndex) -> None:
    index_path = docs_dir / INDEX_FILENAME
    payload = {
        "vectorizer": index.vectorizer,
        "matrix": index.matrix,
        "chunks": index.chunks,
        "sources": index.sources,
    }
    with index_path.open("wb") as f:
        pickle.dump(payload, f)


def _read_file_to_text(path: Path) -> str:
    """Read a text/markdown/PDF file into a single text string."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        if PyPDF2 is None:
            return ""
        text_parts: List[str] = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    continue
        return "\n".join(text_parts)
    return ""


def _chunk_text(text: str, chunk_size: int = 600) -> List[str]:
    """Naively chunk text into smaller pieces based on character count."""
    text = text.replace("\r", " ")
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size) if text[i : i + chunk_size].strip()]


def index_uploaded_file(uploaded_file, docs_dir: Path, tool_log: List[str] | None = None) -> None:
    """Index an uploaded file into the TF-IDF store.

    Args:
        uploaded_file: A Streamlit UploadedFile instance.
        docs_dir: Directory where document copies and the index are stored.
        tool_log: Optional list to append log messages to.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest_path = docs_dir / uploaded_file.name
    with dest_path.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    text = _read_file_to_text(dest_path)
    if not text.strip():
        if tool_log is not None:
            tool_log.append(f"RAG: Skipped {uploaded_file.name} (no readable text).")
        return

    chunks = _chunk_text(text)
    sources = [uploaded_file.name] * len(chunks)

    existing = _load_index(docs_dir)
    if existing is None:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(chunks)
        index = RagIndex(
            vectorizer=vectorizer,
            matrix=matrix,
            chunks=chunks,
            sources=sources,
        )
    else:
        all_chunks = existing.chunks + chunks
        all_sources = existing.sources + sources
        vectorizer = existing.vectorizer
        matrix = vectorizer.fit_transform(all_chunks)
        index = RagIndex(
            vectorizer=vectorizer,
            matrix=matrix,
            chunks=all_chunks,
            sources=all_sources,
        )

    _save_index(docs_dir, index)
    if tool_log is not None:
        tool_log.append(f"RAG: Indexed {uploaded_file.name} with {len(chunks)} chunks.")


def list_indexed_docs(docs_dir: Path) -> List[str]:
    """List filenames of indexed documents in the docs_dir."""
    if not docs_dir.exists():
        return []
    return sorted(
        {p.name for p in docs_dir.iterdir() if p.is_file() and p.name != INDEX_FILENAME}
    )


def query_index(
    query: str,
    docs_dir: Path,
    top_k: int = 3,
) -> List[Tuple[str, str]]:
    """Query the TF-IDF index and return top matching chunks.

    Args:
        query: User query string.
        docs_dir: Directory where the index is stored.
        top_k: Number of chunks to retrieve.

    Returns:
        List of (source_filename, text_chunk) tuples.
    """
    index = _load_index(docs_dir)
    if index is None or not index.chunks:
        return []

    q_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index.matrix)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    results: List[Tuple[str, str]] = []
    for idx in top_indices:
        results.append((index.sources[idx], index.chunks[idx]))
    return results
