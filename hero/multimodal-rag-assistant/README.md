# Multimodal RAG Assistant (Image + Document Q&A)

This app performs multimodal retrieval-augmented generation by combining:

- Document retrieval (FAISS + MiniLM embeddings)
- Image understanding (BLIP captioning)
- LLM reasoning (OpenAI / Ollama / Placeholder)

## Features
- Upload PDFs or TXT files
- Upload images (jpg/png)
- Ask a question grounded in BOTH sources
- Real multimodal reasoning

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
