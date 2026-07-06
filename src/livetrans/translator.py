"""OpenAI-compatible translation client with editable prompt templates."""

from __future__ import annotations

import collections
from typing import Any

from .llm import DEFAULT_LLM_BASE_URL, LLMClient, LLMError
from .prompt_loader import render_prompt

__all__ = [
    "build_prompt",
    "clean_translation_lines",
    "OpenAICompatibleTranslator",
]


def build_prompt(
    texts: list[str],
    context_text: str,
    game_hint: str,
    prompt_extra: str,
    streamer_type: str = "Vtuber",
    streamer_name: str = "鹿乃",
) -> str:
    input_text = "\n".join(texts)
    context_block = ""
    if context_text:
        context_block = (
            "【前文对话上下文】（用于理解语境，不要翻译这部分）：\n"
            f"{context_text}\n"
        )
    return render_prompt(
        "translation.txt",
        streamer_type=streamer_type or "Vtuber",
        streamer_name=streamer_name or "鹿乃",
        game_hint=game_hint,
        prompt_extra=prompt_extra,
        context_block=context_block,
        input_text=input_text,
    )


def clean_translation_lines(content: str, expected: int) -> list[str]:
    clean: list[str] = []
    for line in content.split("\n"):
        line = line.strip()
        if len(line) > 2 and line[0].isdigit() and line[1] in (".", "、"):
            line = line[2:].strip()
        if line:
            clean.append(line)
    while len(clean) < expected:
        clean.append("...")
    return clean


class OpenAICompatibleTranslator:
    def __init__(self, max_context_buffer: int = 20):
        self.context_buffer: "collections.deque[dict[str, str]]" = collections.deque(
            maxlen=max_context_buffer
        )

    def _build_context_text(self, use_context: bool, window: int) -> str:
        if not use_context or not self.context_buffer:
            return ""
        recent = list(self.context_buffer)[-window:]
        lines: list[str] = []
        for ctx in recent:
            lines.append(f"原文: {ctx['orig']}")
            lines.append(f"译文: {ctx['tran']}")
        return "\n".join(lines)

    def translate(self, texts: list[str], config: dict[str, Any]) -> list[str]:
        if not texts or texts == ["无"]:
            return ["..."] * len(texts)
        api_key = config.get("llm_api_key", "")
        base_url = config.get("llm_base_url") or DEFAULT_LLM_BASE_URL
        model = config.get("llm_model", "gpt-4.1-mini")
        client = LLMClient(base_url=base_url, api_key=api_key, model=model)
        if client.requires_api_key and not api_key:
            return ["【未配置Key】"] * len(texts)

        use_context = config.get("use_translation_context", True)
        window = config.get("context_window_size", 5)
        context_text = self._build_context_text(use_context, window)

        prompt = build_prompt(
            texts,
            context_text,
            config.get("game_hint", ""),
            config.get("prompt_extra", ""),
            config.get("streamer_type", "Vtuber"),
            config.get("streamer_name", "鹿乃"),
        )
        try:
            content = client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        except LLMError:
            return ["(失败)"] * len(texts)

        clean = clean_translation_lines(content, len(texts))
        if use_context:
            for i, orig in enumerate(texts):
                self.context_buffer.append(
                    {"orig": orig, "tran": clean[i] if i < len(clean) else ""}
                )
        return clean
