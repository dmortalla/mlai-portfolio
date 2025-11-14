import streamlit as st
from rag_pipeline import load_documents, build_faiss_index, answer_query
from llm_backends import select_backend

st.set_page_config(page_title="RAG Document Advisor", layout="wide")
st.title("📄 AI Document Advisor (RAG Pipeline)")

st.sidebar.header("LLM Backend")
backend_choice = st.sidebar.selectbox(
    "Choose backend:",
    ["OpenAI", "Ollama", "Placeholder"]
)

openai_key = None
if backend_choice == "OpenAI":
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")

st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (.pdf or .txt)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} document(s) loaded.")
    documents = load_documents(uploaded_files)
    faiss_index, chunks = build_faiss_index(documents)
else:
    faiss_index, chunks = None, None

st.markdown("### Ask a question about your documents")
user_question = st.text_input("Your question:")

if st.button("Get Answer") and faiss_index is not None:
    llm = select_backend(backend_choice, api_key=openai_key)
    answer, retrieved_chunks = answer_query(
        user_question,
        faiss_index,
        chunks,
        llm
    )

    st.markdown("### 🔍 Answer")
    st.write(answer)

    st.markdown("### 📚 Retrieved Context")
    for c in retrieved_chunks:
        st.markdown(f"> {c}")

elif st.button("Get Answer"):
    st.warning("Please upload documents first.")

