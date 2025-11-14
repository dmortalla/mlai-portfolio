from openai import OpenAI
import requests

class PlaceholderLLM:
    def generate(self, prompt):
        return "Placeholder response (use OpenAI or Ollama for real output)."

def select_backend(name, api_key=None):
    if name == "OpenAI":
        return OpenAILLM(api_key)
    elif name == "Ollama":
        return OllamaLLM()
    return PlaceholderLLM()

class OpenAILLM:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]

class OllamaLLM:
    def generate(self, prompt):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt}
        )
        return response.json().get("response", "")
