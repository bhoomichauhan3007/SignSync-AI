"""LLM providers behind one small interface: generate(system_prompt, user_prompt) -> text.

Providers:
- extractive (default): no LLM at all. Handled by assistant.py, which returns retrieved passages.
- gemini: Google Gemini REST API. Needs GEMINI_API_KEY in the environment. Uses only the
  standard library (urllib), so it adds no dependency. Free-tier quotas are set by Google and
  change over time; a 429 response means the quota or rate limit was hit.
- ollama: a local model served by Ollama at OLLAMA_URL. No key, no quota, no internet.

Error messages never include the API key.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


class LLMError(Exception):
    """The LLM could not produce an answer (quota, network, bad response). Safe to show to users."""


class LLMConfigError(LLMError):
    """The provider is not configured (for example, the API key is missing)."""


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


def _gemini_http_message(code: int) -> str:
    if code == 429:
        return ("The Gemini free-tier quota or rate limit has been reached. Wait and try again, "
                "or set RAG_LLM_PROVIDER=extractive.")
    if code in (401, 403):
        return "The Gemini API key is invalid or the Generative Language API is not enabled for it."
    if code == 404:
        return "The Gemini model name was not found. Check GEMINI_MODEL."
    if code == 400:
        return "Gemini rejected the request. Check GEMINI_MODEL and the request size."
    return f"Gemini API returned HTTP {code}."


def _extract_gemini_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        reason = (payload.get("promptFeedback") or {}).get("blockReason", "unknown")
        raise LLMError(f"Gemini returned no answer (block reason: {reason}).")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise LLMError("Gemini returned an empty answer.")
    return text


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str | None, model: str, timeout: float = 30.0):
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set. Add it to backend/rag/.env or export it.")
        self._api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{GEMINI_ENDPOINT}/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError(_gemini_http_message(exc.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise LLMError("Could not reach the Gemini API (network problem or timeout).") from None
        except json.JSONDecodeError:
            raise LLMError("Gemini returned a response that could not be read.") from None
        return _extract_gemini_text(payload)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError):
            raise LLMError(f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running?") from None
        except json.JSONDecodeError:
            raise LLMError("Ollama returned a response that could not be read.") from None
        text = ((payload.get("message") or {}).get("content") or "").strip()
        if not text:
            raise LLMError("Ollama returned an empty answer.")
        return text


def build_provider(config) -> LLMProvider | None:
    """Return the configured provider, or None for extractive (no-LLM) mode."""
    choice = config.llm_provider
    if choice == "auto":
        # Gemini when a key is present, otherwise no LLM (the tutor shows the matching passages).
        choice = "gemini" if config.gemini_api_key else "extractive"
    if choice == "extractive":
        return None
    if choice == "gemini":
        return GeminiProvider(config.gemini_api_key, config.gemini_model, config.llm_timeout_seconds)
    if choice == "ollama":
        return OllamaProvider(config.ollama_url, config.ollama_model, config.llm_timeout_seconds)
    raise LLMConfigError(f"Unknown RAG_LLM_PROVIDER '{choice}'. Use auto, extractive, gemini or ollama.")
