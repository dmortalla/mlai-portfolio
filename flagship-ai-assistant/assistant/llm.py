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
            "You are a personal AI assistant that is concise, honest, and helpful. "
            "If you reference external documents, clearly indicate when you are doing so."
        )
        if persona == "Explain like I'm 12":
            return base + " Always explain things simply, as if talking to a 12-year-old."
        if persona == "Technical AI mentor":
            return base + " You are mentoring an aspiring ML/AI engineer; be precise and practical."
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
                "LLM API key is not configured, so I cannot call a real model.

"
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
