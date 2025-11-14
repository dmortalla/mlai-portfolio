"""Streamlit front-end for the Personal AI Assistant.

This app provides:
- A chat interface.
- Document upload & indexing for RAG.
- A simple view of stored long-term memory.
- A debug view of tool calls.

The heavy lifting (LLM calls, RAG, memory, tools, routing) is handled
by the modules inside the `assistant` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

from assistant.router import route_message
from assistant.rag import index_uploaded_file, list_indexed_docs
from assistant.memory import load_memory_snapshot


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"


def _ensure_dirs() -> None:
    """Ensure the data/document directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    state_defaults = {
        "messages": [],          # chat history: list of dicts with role/content
        "tool_log": [],          # list of strings describing tool calls
        "persona": "Helpful general assistant",
    }
    for key, default in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_sidebar() -> None:
    """Render the left-hand sidebar with controls."""
    st.sidebar.header("⚙️ Assistant Settings")
    st.session_state.persona = st.sidebar.selectbox(
        "Assistant persona",
        options=[
            "Helpful general assistant",
            "Explain like I'm 12",
            "Technical AI mentor",
        ],
        index=["Helpful general assistant", "Explain like I'm 12", "Technical AI mentor"].index(
            st.session_state.persona
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Document Upload for RAG")
    uploaded_files = st.sidebar.file_uploader(
        "Upload text / markdown / PDF files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            with st.spinner(f"Indexing {f.name}..."):
                index_uploaded_file(f, DOCS_DIR, tool_log=st.session_state.tool_log)
        st.sidebar.success("Documents indexed. You can now ask questions about them.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Indexed Documents")
    docs = list_indexed_docs(DOCS_DIR)
    if docs:
        for d in docs:
            st.sidebar.write(f"- {d}")
    else:
        st.sidebar.caption("No indexed documents yet.")


def render_chat_tab() -> None:
    """Render the main chat interface tab."""
    st.header("💬 Personal AI Assistant")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything, or reference your documents...")
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Route to assistant
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = route_message(
                    user_input,
                    persona=st.session_state.persona,
                    conversation_history=st.session_state.messages,
                    tool_log=st.session_state.tool_log,
                    docs_dir=DOCS_DIR,
                )
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def render_memory_tab() -> None:
    """Render a read-only snapshot of long-term memory."""
    st.header("🧠 Long-Term Memory Snapshot")
    memory = load_memory_snapshot(DATA_DIR)
    if not memory:
        st.info("No long-term memory stored yet.")
        return

    st.json(memory)


def render_tools_tab() -> None:
    """Render a simple log of tool invocations."""
    st.header("🛠️ Tool Call Log")
    if not st.session_state.tool_log:
        st.info("No tools have been called yet.")
        return

    for entry in st.session_state.tool_log:
        st.markdown(f"- {entry}")


def main() -> None:
    """Run the Streamlit Personal AI Assistant app."""
    _ensure_dirs()
    st.set_page_config(page_title="Personal AI Assistant", layout="wide")
    init_session_state()

    tab_chat, tab_memory, tab_tools = st.tabs(["Chat", "Memory", "Tools"])

    with tab_chat:
        render_chat_tab()

    with tab_memory:
        render_memory_tab()

    with tab_tools:
        render_tools_tab()


if __name__ == "__main__":
    main()
