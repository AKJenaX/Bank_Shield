from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    raw: Dict[str, Any]


class _ChatCompletions:
    def __init__(self, parent: "LLMClient") -> None:
        self._parent = parent

    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        return self._parent.chat_completions_create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


class _Chat:
    def __init__(self, parent: "LLMClient") -> None:
        self.completions = _ChatCompletions(parent)


class LLMClient:
    """
    Minimal OpenAI chat-completions client using the OpenAI SDK.

    - Reads auth token from env (OPENAI_API_KEY preferred)
    - Uses API_BASE_URL only as optional base URL override
    - Exposes `client.chat.completions.create(...)` for reuse
    """

    def __init__(
        self,
        *,
        base_url: Optional[str],
        api_key: str,
        timeout_s: float = 60.0,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._base_url = (base_url or "").strip() or None
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._default_headers = default_headers or {}
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_s,
            default_headers=self._default_headers or None,
        )
        self.chat = _Chat(self)

    def chat_completions_create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as e:
            raise LLMClientError(f"LLM request failed: {e}") from e

        text = ""
        try:
            text = (resp.choices or [])[0].message.content or ""
        except Exception:
            text = ""
        return ChatCompletionResult(text=text or "", raw=resp.model_dump())


def get_llm_client() -> LLMClient:
    """
    Create an LLM client using only environment variables:
      - OPENAI_API_KEY (preferred)
      - API_BASE_URL (optional, for compatible providers/proxies)
    """

    base_url = (os.getenv("API_BASE_URL") or "http://localhost:11434/v1").strip() or None
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("HF_TOKEN")
        or "ollama"
    ).strip()
    if not api_key:
        raise ValueError("Missing required env var: OPENAI_API_KEY")

    timeout_s = float(os.getenv("LLM_TIMEOUT_S") or "60")
    return LLMClient(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
