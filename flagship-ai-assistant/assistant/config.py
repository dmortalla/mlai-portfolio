"""Configuration utilities for the Personal AI Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application-level settings.

    Attributes:
        openai_api_key: API key for OpenAI. If missing, the app will fall back
            to a simple, non-LLM placeholder response.
        model_name: Name of the OpenAI chat model.
    """

    openai_api_key: str | None
    model_name: str = "gpt-4o-mini"


def get_settings() -> Settings:
    """Load settings from environment variables.

    Returns:
        Settings: Resolved settings object.
    """
    api_key = os.getenv("OPENAI_API_KEY") or None
    return Settings(openai_api_key=api_key)
