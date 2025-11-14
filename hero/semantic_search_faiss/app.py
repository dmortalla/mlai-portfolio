"""Streamlit app for a semantic search engine using FAISS and sentence-transformers.

This app lets you:
- Upload a CSV or TXT file with text data.
- Build a vector index using sentence embeddings.
- Run semantic search queries over your data.
- Optionally use an LLM (OpenAI / Ollama / placeholder) to summarize the top results.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from faiss_search import build_index_from_texts, search_similar_texts
from llm_backends import select_backend


def main() -> None:
    """Run the Streamlit semantic search application.

    The function configures the Streamlit page, loads user data, builds a FAISS
    index if text is provided, and then allows the user to run semantic search
    queries with optional LLM summaries.
    """
    st.set_page_config(page_title="Semantic Search Engine", layout="wide")
    st.title("🔍 Semantic Search Engine (FAISS + Embeddings)")

    st.markdown(
        "Upload a CSV or TXT file, pick the text column, and run semantic search "
        "over your data using sentence embeddings and FAISS."
    )

    # Sidebar: LLM backend settings (optional)
    st.sidebar.header("LLM Backend (Optional)")
    backend_choice = st.sidebar.selectbox(
        "Use LLM to summarize results?", ["None", "OpenAI", "Ollama", "Placeholder"], index=0
    )

    openai_key: Optional[str] = None
    if backend_choice == "OpenAI":
        openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

    # Sidebar: file upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV or TXT file",
        type=["csv", "txt"],
    )

    text_data = []
    df = None

    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            # Load CSV into DataFrame
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as exc:  # guardrail: handle bad CSVs
                st.error(f"Error reading CSV file: {exc}")
                return

            st.subheader("Preview of Uploaded CSV")
            st.dataframe(df.head())

            # Let the user choose which text column to search
            text_columns = df.select_dtypes(include=["object"]).columns.tolist()
            if not text_columns:
                st.error("No text columns found in this CSV. Please upload a valid file.")
                return

            text_col = st.selectbox(
                "Select the text column to index:",
                text_columns,
            )

            text_data = df[text_col].fillna("").astype(str).tolist()

        else:
            # TXT file: one line per record
            try:
                content = uploaded_file.read().decode("utf-8", errors="ignore")
            except Exception as exc:  # guardrail: handle decoding failure
                st.error(f"Error reading TXT file: {exc}")
                return
            text_data = [line.strip() for line in content.splitlines() if line.strip()]

            st.subheader("Preview of Uploaded TXT")
            st.write("\n".join(text_data[:10]))

    # Build FAISS index if we have text
    index = None
    metadata = None

    if text_data:
        with st.spinner("Building FAISS index over text data..."):
            try:
                index, metadata = build_index_from_texts(text_data)
            except ValueError as exc:
                st.error(str(exc))
                return
        st.success(f"Index built with {len(text_data)} entries.")
    elif uploaded_file is not None:
        st.warning("No valid text extracted from file to build an index.")

    # Query section
    st.markdown("---")
    st.subheader("Run Semantic Search"")

    query = st.text_input("Enter your search query:")
    top_k = st.slider("Top-k results", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        if index is None or metadata is None:
            st.warning("Please upload a file and build the index first.")
            return

        if not query.strip():
            st.warning("Please enter a non-empty query.")
            return

        with st.spinner("Searching..."):
            try:
                results = search_similar_texts(query, index, metadata, top_k=top_k)
            except ValueError as exc:
                st.error(str(exc))
                return

        if not results:
            st.info("No results found for this query.")
            return

        st.markdown("### 🔎 Top Results")
        for i, item in enumerate(results, start=1):
            st.markdown(f"**Result {i}** (distance: {item['score']:.4f})")
            st.markdown(f"> {item['text']}")
            st.markdown("---")


        if backend_choice != "None":
            # Optional LLM summary
            try:
                llm = select_backend(backend_choice, api_key=openai_key)
            except ValueError as exc:
                st.error(str(exc))
                return

            joined_context = "\n\n".join([r["text"] for r in results])

            prompt = (
                "You are a helpful assistant. The user ran a semantic search.\n"
                "Here are the top matching passages:\n\n"
                f"{joined_context}\n\n"
                f"User query: {query}\n\n"
                "Provide a concise, helpful summary that answers the user based on this context."
            )

            with st.spinner("Asking LLM to summarize results..."):
                summary = llm.generate(prompt)

            st.markdown("### 🤖 LLM Summary (Optional)")
            st.write(summary)


if __name__ == "__main__":
    main()
