"""Simple LLM backend selector.

Supports:
- OpenAI
- Ollama
- Placeholder (offline stub)
"""

from typing import Optional

from openai import OpenAI
import requests


class PlaceholderLLM:
    """Offline placeholder backend used when no real LLM is configured."""

    def generate(self, prompt: str) -> str:
        """Return a simple canned response.

        Args:
            prompt: The text prompt (ignored).

        Returns:
            A placeholder response string.
        """
        return (
            "This is a placeholder response. Configure OpenAI or Ollama for real LLM answers."
        )


class OpenAILLM:
    """LLM backend using the OpenAI Chat Completions API."""

    def __init__(self, api_key: Optional[str]) -> None:
        """Initialize the OpenAI client.

        Args:
            api_key: The OpenAI API key.

        Raises:
            ValueError: If the API key is missing.
        """
        if not api_key:
            raise ValueError("OpenAI API key is required for the OpenAI backend.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        """Generate a response using an OpenAI chat model.

        Args:
            prompt: Input text prompt.

        Returns:
            Model-generated response text.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        # Note: new OpenAI client returns a Message object; access its 'content' field.
        return response.choices[0].message["content"]  # type: ignore[index]


class OllamaLLM:
    """LLM backend using a local Ollama server."""

    def generate(self, prompt: str) -> str:
        """Generate a response using a local Ollama model.

        Args:
            prompt: Input text prompt.

        Returns:
            Model-generated response text (or error message).
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3", "prompt": prompt},
                timeout=60,
            )
            data = response.json()
            return data.get("response", "")
        except Exception as exc:  # guardrail: catch connection issues
            return f"Error: Could not reach Ollama server ({exc}). Is it running?"


def select_backend(name: str, api_key: Optional[str] = None):
    """Return an LLM backend instance based on the selected name.

    Args:
        name: Backend name: 'OpenAI', 'Ollama', or 'Placeholder'/'None'.
        api_key: Optional API key for OpenAI.

    Returns:
        An instance of an LLM backend class.
    """
    if name == "OpenAI":
        return OpenAILLM(api_key)
    if name == "Ollama":
        return OllamaLLM()
    return PlaceholderLLM()
