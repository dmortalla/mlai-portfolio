"""Routing logic that decides how to handle user messages.

The router decides whether to:
- Call the LLM directly.
- Perform a RAG lookup and pass context to the LLM.
- Call a tool (e.g., calculator) and blend its result into the answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from .llm import call_llm
from .rag import query_index
from .memory import update_memory


def _maybe_extract_math_expr(message: str) -> str | None:
    """Extract a math expression if the user clearly asks for a calculation."""
    lowered = message.lower()
    if any(keyword in lowered for keyword in ["calculate", "compute", "eval", "evaluate"]):
        # naive heuristic: take everything after the keyword
        for kw in ["calculate", "compute", "eval", "evaluate"]:
            if kw in lowered:
                idx = lowered.index(kw) + len(kw)
                expr = message[idx:].strip(": ,.")
                return expr or None
    return None


def _maybe_rag_query(message: str, docs_dir: Path) -> list[tuple[str, str]]:
    """Decide whether to run a RAG query based on the message content."""
    lowered = message.lower()
    triggers = ["in my documents", "from my docs", "based on the files", "search my docs"]
    if any(t in lowered for t in triggers):
        return query_index(message, docs_dir, top_k=3)
    # Fallback: if we have any index at all and message is a question, we can still try
    if "?" in message:
        return query_index(message, docs_dir, top_k=2)
    return []


def _build_llm_messages(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    rag_results: list[tuple[str, str]] | None,
    tool_result: str | None,
) -> List[Dict[str, str]]:
    """Build the messages list to send to the LLM."""
    messages: List[Dict[str, str]] = []

    if rag_results:
        context_chunks = "\n\n".join(
            f"From {src} -> {chunk}" for src, chunk in rag_results
        )
        messages.append(
            {
                "role": "system",
                "content": (
                    "You have access to the following document excerpts. "
                    "Use them if relevant, and clearly cite that you are "
                    "using document context.\n\n" + context_chunks
                ),
            }
        )

    if tool_result is not None:
        messages.append(
            {
                "role": "system",
                "content": (
                    "A calculation tool has returned the following result: "
                    f"{tool_result}. Use this in your answer if helpful."
                ),
            }
        )

    # Append last few turns of the conversation for continuity
    for msg in conversation_history[-6:]:
        messages.append(msg)

    # Ensure the latest user message is present
    if not messages or messages[-1].get("role") != "user":
        messages.append({"role": "user", "content": user_message})

    return messages


def route_message(
    user_message: str,
    persona: str,
    conversation_history: List[Dict[str, str]],
    tool_log: List[str],
    docs_dir: Path,
) -> str:
    """Route a user message to the appropriate components.

    This is the main entry point the Streamlit app calls.

    Args:
        user_message: Raw user message from the chat UI.
        persona: Selected assistant persona.
        conversation_history: List of messages so far.
        tool_log: Mutable list where tool actions are logged.
        docs_dir: Directory containing RAG documents and index.

    Returns:
        The assistant's reply text.
    """
    from .tools import safe_calculate  # local import to avoid cycles

    tool_result: str | None = None

    # 1) Try to detect math expression
    expr = _maybe_extract_math_expr(user_message)
    if expr:
        try:
            value = safe_calculate(expr)
            tool_result = f"{expr} = {value}"
            tool_log.append(f"Calculator: evaluated '{expr}' -> {value}")
        except Exception as exc:
            tool_log.append(f"Calculator error on '{expr}': {exc}")

    # 2) Try RAG if any docs exist
    rag_results: list[tuple[str, str]] | None = None
    try:
        rag_hits = _maybe_rag_query(user_message, docs_dir)
        if rag_hits:
            tool_log.append(f"RAG: retrieved {len(rag_hits)} chunks for query.")
            rag_results = rag_hits
    except Exception as exc:  # guardrail
        tool_log.append(f"RAG query failed: {exc}")

    # 3) Build messages for LLM
    messages = _build_llm_messages(
        user_message=user_message,
        conversation_history=conversation_history,
        rag_results=rag_results,
        tool_result=tool_result,
    )

    # 4) Optionally update memory: if user states a clear preference
    if "i prefer" in user_message.lower():
        update_memory(
            docs_dir.parent,
            key="user_preference_last_statement",
            value=user_message,
        )

    # 5) Call LLM
    reply = call_llm(messages, persona=persona)
    return reply
