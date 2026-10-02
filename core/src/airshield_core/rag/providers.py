"""Language-model providers.

The assistant needs an LLM only to *phrase* an answer from material that has
already been retrieved and from values AirShield already computed. It is never
asked to produce a number.

Two providers are supported:

* :class:`OpenAICompatibleProvider` - any ``/chat/completions`` endpoint: OpenAI,
  a self-hosted vLLM/Ollama gateway, or an internal proxy. Configured entirely by
  environment variables.
* :class:`NoLLMProvider` - the honest default when nothing is configured. It
  reports itself unavailable so the answerer falls back to returning retrieved
  source excerpts instead of pretending a model answered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


class LLMError(RuntimeError):
    """Raised when a provider is configured but the call fails."""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    """A chat completion backend."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def is_available(self) -> bool: ...

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResult: ...


class NoLLMProvider:
    """Placeholder used when no LLM is configured.

    ``is_available`` is False and ``complete`` raises. This is what keeps the
    system honest: with no model configured, the assistant answers with retrieved
    source text and says so, rather than inventing prose.
    """

    def __init__(self, reason: str = "no language model is configured"):
        self._reason = reason

    @property
    def name(self) -> str:
        return "none"

    @property
    def model(self) -> str:
        return ""

    @property
    def is_available(self) -> bool:
        return False

    @property
    def reason(self) -> str:
        return self._reason

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResult:
        raise LLMError(self._reason)


class OpenAICompatibleProvider:
    """Calls any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 45.0,
        extra_headers: dict[str, str] | None = None,
    ):
        if not base_url:
            raise LLMError("base_url is required")
        if not model:
            raise LLMError("model is required")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout
        self._extra_headers = extra_headers or {}

    @property
    def name(self) -> str:
        return f"openai-compatible:{self._base_url}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResult:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach the language model: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:300]
            raise LLMError(
                f"language model returned HTTP {response.status_code}: {detail}"
            )

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected language model response shape: {exc}") from exc

        if not isinstance(text, str) or not text.strip():
            raise LLMError("language model returned an empty completion")

        return LLMResult(text=text.strip(), provider=self.name, model=self._model)


def build_provider(settings) -> LLMProvider:
    """Build the provider described by settings, or the no-op provider.

    Nothing is guessed: with no base URL or model configured, the no-op provider
    is returned and the answerer degrades to retrieval-only mode.
    """
    base_url = getattr(settings, "llm_base_url", "") or ""
    model = getattr(settings, "llm_model", "") or ""
    if not base_url or not model:
        return NoLLMProvider(
            "no language model is configured (set AIRSHIELD_LLM_BASE_URL and "
            "AIRSHIELD_LLM_MODEL to enable generated explanations)"
        )

    extra_headers: dict[str, str] = {}
    raw_headers = getattr(settings, "llm_extra_headers", "") or ""
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
            if isinstance(parsed, dict):
                extra_headers = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            extra_headers = {}

    return OpenAICompatibleProvider(
        base_url=base_url,
        model=model,
        api_key=getattr(settings, "llm_api_key", "") or "",
        timeout=float(getattr(settings, "llm_timeout", 45.0)),
        extra_headers=extra_headers,
    )
