import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from captioner import caption_image

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------------
# Load Documents
# -------------------------------------------------
def load_documents(uploaded_docs):
    all_chunks = []

    for f in uploaded_docs:
        if f.name.endswith(".pdf"):
            with pdfplumber.open(f) as pdf:
                text = "\n".join([p.extract_text() or "" for p in pdf.pages])
        else:
            text = f.read().decode("utf-8")

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        all_chunks.extend(chunks)

    return all_chunks


# -------------------------------------------------
# Build FAISS Index
# -------------------------------------------------
def build_faiss_index(chunks):
    if not chunks:
        return None, None

    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, chunks


# -------------------------------------------------
# Extract Image Caption/Text
# -------------------------------------------------
def extract_image_context(file):
    """Generate a caption from the image."""
    return caption_image(file)


# -------------------------------------------------
# Multimodal Query Answering
# -------------------------------------------------
def answer_query_multimodal(query, index, chunks, image_context, llm):
    doc_context = []

    # Search documents if available
    if index is not None:
        query_embedding = embedder.encode([query])
        k = 3
        scores, indices = index.search(np.array(query_embedding), k)
        doc_context = [chunks[i] for i in indices[0]]

    # Combine contexts
    combined_context = ""

    if doc_context:
        combined_context += "Document Context:\n" + "\n".join(doc_context)

    if image_context:
        combined_context += f"\n\nImage Context:\n{image_context}"

    prompt = f"""
You are a multimodal assistant.

Use the following context (from documents + image):

{combined_context}

Question: {query}

Answer:
    """

    answer = llm.generate(prompt)
    return answer, doc_context
