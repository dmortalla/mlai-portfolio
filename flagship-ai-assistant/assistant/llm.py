"""LLM utilities for the Personal AI Assistant.

This module wraps OpenAI's Chat Completions API behind a simple function
that can gracefully degrade to a placeholder implementation if no API key
is configured.
"""

from __future__ import annotations

from typing import List, Dict

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .config import get_settings


def _build_system_prompt(persona: str) -> str:
    """Build a system prompt string based on the selected persona."""
    base = (
        "You are the assistant inside a modular AI application. "
        "This application includes a Streamlit chat interface, a router that orchestrates requests, "
        "a TF-IDF retrieval component for uploaded documents, a JSON-based memory component for simple user preferences, "
        "and tool support including a safe calculator. "
        "When asked about your capabilities, explain them in terms of this application architecture rather than as a generic LLM. "
        "Be concise, honest, and helpful. "
        "If retrieved document context is used, say so explicitly. "
        "Do not claim to have used tools, memory, or retrieval unless they were actually part of the current interaction."
    )
    if persona == "Explain like I'm 12":
        return base + " Always explain things simply, as if talking to a 12-year-old."
    if persona == "Technical AI mentor":
        return base + " You are mentoring an aspiring ML/AI engineer; be precise, practical, and system-oriented."
    return base


def call_llm(
    messages: List[Dict[str, str]],
    persona: str,
) -> str:
    """Call the LLM (OpenAI) or return a safe fallback.

    Args:
        messages: Conversation history as a list of role/content dicts.
        persona: Selected assistant persona.

    Returns:
        The assistant's reply text.
    """
    settings = get_settings()
    system_prompt = _build_system_prompt(persona)

    if settings.openai_api_key is None or OpenAI is None:
        # Fallback: no external call; echo with a simple prefix.
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return (
            "LLM API key is not configured, so I cannot call a real model."
            f"(Echoing your last message for demo purposes): {last_user}"
        )

    client = OpenAI(api_key=settings.openai_api_key)

    chat_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=chat_messages,
        )
    except Exception as exc:  # guardrail
        return f"LLM call failed: {exc}"

    choice = response.choices[0]
    return choice.message.content or ""
