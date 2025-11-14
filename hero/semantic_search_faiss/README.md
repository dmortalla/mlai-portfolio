# Semantic Search Engine – FAISS + Sentence Embeddings

This app lets you upload a CSV or TXT file and run semantic search over a chosen text column using:

- `sentence-transformers` (MiniLM embeddings)
- `faiss-cpu` for vector similarity search
- Optional LLM (OpenAI / Ollama / Placeholder) to summarize results

## 🚀 Features

- Upload text data (CSV or TXT)
- Build a FAISS index over sentence embeddings
- Run top-k semantic search
- View scored results
- Optionally get an LLM-based summary of the results

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

- Works well on Streamlit Cloud (CPU only).
- Can be mirrored on Hugging Face Spaces if desired.
