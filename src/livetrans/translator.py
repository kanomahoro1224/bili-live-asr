"""OpenAI 兼容翻译客户端 + 上下文记忆（仅依赖 requests）。

逐行日译中，保持行数对应、口语化、去标点；带上下文窗口（最近 N 句原/译文）
帮助理解代词与省略。通过 llm_base_url 可接入任意 OpenAI 兼容服务。
build_prompt / clean_translation_lines 为纯函数，可独立单测。
"""

from __future__ import annotations

import collections
from typing import Any

from .llm import DEFAULT_LLM_BASE_URL, LLMClient, LLMError

__all__ = [
    "build_prompt",
    "clean_translation_lines",
    "OpenAICompatibleTranslator",
]


def build_prompt(
    texts: list[str], context_text: str, game_hint: str, prompt_extra: str
) -> str:
    """组装翻译提示词。context_text 为空走无上下文分支。"""
    input_text = "\n".join(texts)
    head = (
        f"现在是一个vtb(鹿乃)在玩/或[{game_hint}]。请将下面的日文句子逐行翻译成中文。"
        "要求：严格保持行数对应，一行日文对应一行中文，不要合并句子，口语化，去掉标点符号。\n"
    )
    if context_text:
        return (
            head
            + f"{prompt_extra}\n\n"
            "【前文对话上下文】（用于理解语境，不要翻译这部分）：\n"
            f"{context_text}\n\n"
            "【待翻译内容】（只翻译这部分，根据上下文理解代词和省略的内容）：\n"
            f"{input_text}"
        )
    return head + f"{prompt_extra}\n待翻译内容：\n{input_text}"


def clean_translation_lines(content: str, expected: int) -> list[str]:
    """清洗模型输出：去行首编号（1. / 1、），补齐到 expected 行。"""
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
    """持有上下文缓冲区的翻译器。每次 translate() 维护最近对话历史。"""

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
        """翻译一批文本，返回等长译文列表。失败返回占位符，不抛出。"""
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
