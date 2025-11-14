import streamlit as st
from multimodal_pipeline import (
    load_documents,
    build_faiss_index,
    extract_image_context,
    answer_query_multimodal
)
from llm_backends import select_backend

st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")
st.title("🖼️📄 Multimodal RAG Assistant (Text + Vision)")

# -------------------------
# Sidebar - LLM Backend
# -------------------------
st.sidebar.header("LLM Backend")
backend_choice = st.sidebar.selectbox(
    "Choose backend:",
    ["OpenAI", "Ollama", "Placeholder"]
)

openai_key = None
if backend_choice == "OpenAI":
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

# -------------------------
# Upload inputs
# -------------------------
uploaded_docs = st.sidebar.file_uploader(
    "Upload documents (.pdf or .txt)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

uploaded_image = st.sidebar.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# Load docs + build index
# -------------------------
if uploaded_docs:
    documents = load_documents(uploaded_docs)
    faiss_index, chunks = build_faiss_index(documents)
    st.sidebar.success(f"Loaded {len(chunks)} document chunks.")
else:
    faiss_index, chunks = None, None

# -------------------------
# Extract image context
# -------------------------
image_context = None
if uploaded_image:
    st.sidebar.success("Image loaded.")
    image_context = extract_image_context(uploaded_image)

# -------------------------
# User question
# -------------------------
st.markdown("### Ask a question based on the image + documents")
user_question = st.text_input("Your question:")

if st.button("Get Answer"):
    if faiss_index is None and image_context is None:
        st.warning("Please upload an image or documents first.")
    else:
        llm = select_backend(backend_choice, api_key=openai_key)
        answer, doc_context = answer_query_multimodal(
            user_question, faiss_index, chunks, image_context, llm
        )

        st.markdown("### 🔮 Answer")
        st.write(answer)

        if doc_context:
            st.markdown("### 📚 Retrieved Document Context")
            for c in doc_context:
                st.markdown(f"> {c}")

        if image_context:
            st.markdown("### 🖼️ Image Context")
            st.info(image_context)
