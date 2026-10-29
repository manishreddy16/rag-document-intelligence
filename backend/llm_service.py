"""LLM provider abstraction. RAGPipeline only depends on LLMService.generate().

GeminiLLMService calls Google's REST API with the standard library, so no extra
package is needed. To change provider, add another class with generate() and
register it in create_llm_service().
"""
import json
import logging
import random
import time
import urllib.error
import urllib.request
from typing import Callable, Protocol

from .config import Settings
from .exceptions import (LLMConfigError, LLMError, LLMRateLimitError,
                         LLMUnavailableError)

logger = logging.getLogger(__name__)

# Temporary conditions worth retrying. Everything else (400/401/403/404...) is permanent.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


class LLMService(Protocol):
    def generate(self, prompt: str) -> str: ...


class GeminiLLMService:
    def __init__(self, api_key: str, model: str, timeout: int = 30,
                 max_retries: int = 3, backoff_seconds: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        # max_retries=3 -> up to 4 attempts, waiting ~1s, ~2s, ~4s (plus small jitter) between them.
        self.max_retries, self.backoff_seconds, self._sleep = max(0, max_retries), backoff_seconds, sleep

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise LLMConfigError("LLM_API_KEY is not set. Add it to your .env file.")
        if not self.model:
            raise LLMConfigError("LLM_MODEL is not set. Add a model name to your .env file.")
        request = urllib.request.Request(
            GEMINI_URL.format(model=self.model.removeprefix("models/")),
            data=json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key},  # key sent in header, not URL
            method="POST")
        data = self._send_with_retry(request)
        return self._extract_text(data)

    def _send_with_retry(self, request: urllib.request.Request) -> dict:
        """Send the request; retry only transient failures, with exponential backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # must come before URLError (its parent)
                error, retryable = self._http_error(exc), exc.code in RETRYABLE_STATUS
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                error = LLMUnavailableError("Could not reach the LLM service. Check your internet connection.")
                error.__cause__, retryable = exc, True
            except ValueError as exc:
                raise LLMUnavailableError("The LLM service returned an unreadable response.") from exc
            if not retryable or attempt == self.max_retries:
                raise error
            delay = self.backoff_seconds * (2 ** attempt) + random.uniform(0, 0.25 * self.backoff_seconds)
            # Log only the error type and delay: never the request, headers or key.
            logger.warning("Transient LLM failure (%s); retry %d/%d in %.1fs",
                           type(error).__name__, attempt + 1, self.max_retries, delay)
            self._sleep(delay)
        raise AssertionError("unreachable")  # loop always returns or raises

    @classmethod
    def _http_error(cls, exc: urllib.error.HTTPError) -> LLMError:
        message = cls._http_message(exc)
        if exc.code == 429:
            return LLMRateLimitError(message)
        if exc.code in (400, 401, 403, 404):  # bad key / model / request = our config
            return LLMConfigError(message)
        return LLMUnavailableError(message)

    @staticmethod
    def _http_message(exc: urllib.error.HTTPError) -> str:
        detail = ""
        try:
            detail = json.loads(exc.read().decode("utf-8"))["error"]["message"][:200]
        except Exception:
            pass
        hints = {429: "Rate limit or quota reached. Wait and try again.",
                 400: "The request was rejected (check LLM_MODEL).",
                 401: "The API key was rejected.", 403: "The API key was rejected or lacks access.",
                 404: "Model not found (check LLM_MODEL)."}
        base = hints.get(exc.code, f"LLM service error (HTTP {exc.code}).")
        return f"{base} {detail}".strip()

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMUnavailableError(f"The LLM returned no answer (blocked: {reason})." if reason
                                      else "The LLM returned no answer.")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise LLMUnavailableError("The LLM returned an empty answer.")
        return text


class OllamaLLMService:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, timeout: int = 30):
        self.model, self.timeout = model or DEFAULT_OLLAMA_MODEL, timeout

    def generate(self, prompt: str) -> str:
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise LLMRateLimitError("The local Ollama service returned HTTP 429.") from exc
            if exc.code in (400, 404):
                raise LLMConfigError(
                    f"Ollama rejected the request (HTTP {exc.code}); check LLM_MODEL "
                    "and ensure the model is installed.") from exc
            raise LLMUnavailableError(
                f"The local Ollama service returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMUnavailableError(
                "Could not reach the local Ollama service. Ensure Ollama is running.") from exc
        except ValueError as exc:
            raise LLMUnavailableError(
                "The local Ollama service returned an unreadable response.") from exc

        if not isinstance(data, dict):
            raise LLMUnavailableError("The local Ollama service returned an invalid response.")
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError("The LLM returned an empty answer.")
        return text.strip()


def llm_is_configured(settings: Settings) -> bool:
    """Report provider configuration completeness, not whether a service is reachable."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return bool(settings.llm_api_key and settings.llm_model)
    if provider == "ollama":
        return True  # Ollama has a built-in model default; reachability is checked on use.
    return False


def create_llm_service(settings: Settings) -> LLMService:
    if settings.llm_provider.lower() == "gemini":
        return GeminiLLMService(settings.llm_api_key, settings.llm_model,
                                settings.llm_timeout_seconds, settings.llm_max_retries)
    if settings.llm_provider.lower() == "ollama":
        return OllamaLLMService(settings.llm_model or DEFAULT_OLLAMA_MODEL,
                                settings.llm_timeout_seconds)
    raise LLMConfigError(f"Unsupported LLM_PROVIDER '{settings.llm_provider}'.")
