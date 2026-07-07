"""OpenAI-compatible chat client used by translation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

__all__ = ["DEFAULT_LLM_BASE_URL", "LLMError", "LLMTimeoutError", "LLMClient"]

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"


class LLMError(RuntimeError):
    """Raised when the OpenAI-compatible endpoint cannot return chat content."""


class LLMTimeoutError(LLMError):
    """Raised when the OpenAI-compatible endpoint exceeds the request timeout."""


def _chat_url(base_url: str) -> str:
    base = (base_url or DEFAULT_LLM_BASE_URL).strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/chat/completions"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _extract_chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("missing choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LLMError("invalid choice")
    message = choice.get("message")
    if isinstance(message, dict) and "content" in message:
        return _content_to_text(message.get("content")).strip()
    if "text" in choice:
        return _content_to_text(choice.get("text")).strip()
    raise LLMError("missing message.content")


@dataclass
class LLMClient:
    base_url: str = DEFAULT_LLM_BASE_URL
    api_key: str = ""
    model: str = "gpt-4.1-mini"
    timeout: float = 60

    @property
    def chat_url(self) -> str:
        return _chat_url(self.base_url)

    @property
    def requires_api_key(self) -> bool:
        host = urlparse(self.chat_url).hostname or ""
        return host not in {"localhost", "127.0.0.1", "::1"}

    def chat(self, messages: list[dict[str, str]], **params: Any) -> str:
        if self.requires_api_key and not self.api_key:
            raise LLMError("missing api key")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **params,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = requests.post(
                self.chat_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return _extract_chat_content(resp.json())
        except requests.exceptions.Timeout as e:
            raise LLMTimeoutError(str(e)) from e
        except Exception as e:
            raise LLMError(str(e)) from e
