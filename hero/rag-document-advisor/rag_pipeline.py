import pdfplumber
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(uploaded_files):
    text_chunks = []

    for file in uploaded_files:
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        else:
            text = file.read().decode("utf-8")

        # Simple split
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        text_chunks.extend(chunks)

    return text_chunks


def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks


def answer_query(query, index, chunks, llm):
    query_embedding = embedder.encode([query])
    k = 3
    scores, indices = index.search(np.array(query_embedding), k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = llm.generate(prompt)
    return answer, retrieved_chunks
