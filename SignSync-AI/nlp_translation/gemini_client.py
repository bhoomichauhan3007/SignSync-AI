# Member 3: Gemini API connection.
# The key is read from backend/.env. If the key or the google-genai package is missing,
# this module still imports; generate_text() raises a clear error and gemini_nlp falls back
# to the rule-based sentence code, so the backend keeps working without an API key.

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

try:
    from google import genai
except ImportError:  # google-genai is optional
    genai = None

PROJECT_ENV = Path(__file__).resolve().parent.parent / "backend" / ".env"
if load_dotenv is not None:
    load_dotenv(PROJECT_ENV)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=API_KEY) if (genai is not None and API_KEY) else None


def generate_text(prompt, max_tokens=200):
    """Send a prompt to Gemini and return generated text."""
    if client is None:
        raise RuntimeError("Gemini is not configured: set GEMINI_API_KEY in backend/.env and install google-genai")
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"max_output_tokens": max_tokens},
    )
    return response.text.strip()


if __name__ == "__main__":
    print(generate_text("Reply with exactly: Gemini connection successful."))
