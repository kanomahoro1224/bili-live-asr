"""OpenAI-compatible translation client with editable prompt templates."""

from __future__ import annotations

import collections
import json
from typing import Any

from .llm import DEFAULT_LLM_BASE_URL, LLMClient, LLMError, LLMTimeoutError
from .prompt_loader import load_prompt, render_prompt

__all__ = [
    "build_prompt",
    "build_system_prompt",
    "clean_translation_lines",
    "OpenAICompatibleTranslator",
]

LANGUAGE_DISPLAY = {
    "auto": "Japanese",
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "zh_tw": "Traditional Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "hi": "Hindi",
}


def build_prompt(
    texts: list[str],
    context_text: str,
    game_hint: str,
    prompt_extra: str,
    streamer_type: str = "Vtuber",
    streamer_name: str = "鹿乃",
) -> str:
    """Backward-compatible renderer for the editable translation prompt file."""
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
        source_lang="Japanese",
        target_lang="Chinese",
        context=context_text,
    )


def _display_language(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return LANGUAGE_DISPLAY.get(text.lower().replace("-", "_"), text)


def build_system_prompt(
    context_text: str,
    config: dict[str, Any],
) -> str:
    source_lang = _display_language(
        config.get("llm_source_lang")
        or config.get("source_language")
        or config.get("asr_language"),
        "Japanese",
    )
    target_lang = _display_language(
        config.get("llm_target_lang") or config.get("target_language"),
        "Chinese",
    )
    prompt_extra = str(config.get("prompt_extra") or "").strip()
    return render_prompt(
        "translation.txt",
        source_lang=source_lang,
        target_lang=target_lang,
        context=context_text,
        prompt_extra=prompt_extra,
        streamer_type=config.get("streamer_type") or "Vtuber",
        streamer_name=config.get("streamer_name") or "鹿乃",
        game_hint=config.get("game_hint") or "",
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


def _parse_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str) and value.strip():
        try:
            items = json.loads(value)
        except json.JSONDecodeError:
            return []
    else:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _build_qwen_mt_options(config: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "source_lang": config.get("qwen_mt_source_lang") or "Japanese",
        "target_lang": config.get("qwen_mt_target_lang") or "Chinese",
    }
    terms = _parse_json_list(config.get("qwen_mt_terms")) if config.get("qwen_mt_terms_enabled") else []
    if terms:
        options["terms"] = terms
    tm_list = (
        _parse_json_list(config.get("qwen_mt_tm_list"))
        if config.get("qwen_mt_tm_list_enabled")
        else []
    )
    if tm_list:
        options["tm_list"] = tm_list
    domains = str(config.get("qwen_mt_domains") or "").strip() if config.get("qwen_mt_domains_enabled") else ""
    if domains:
        options["domains"] = domains
    return options


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
            lines.append(f"Source: {ctx['orig']}")
            lines.append(f"Translation: {ctx['tran']}")
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def _build_messages(
        self,
        system_prompt: str,
        text: str,
        use_context: bool,
        window: int,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        if use_context and self.context_buffer and "{context}" not in load_prompt("translation.txt"):
            for ctx in list(self.context_buffer)[-window:]:
                messages.append({"role": "user", "content": ctx["orig"]})
                messages.append({"role": "assistant", "content": ctx["tran"]})
        messages.append({"role": "user", "content": text})
        return messages

    def _build_client(
        self,
        config: dict[str, Any],
        *,
        base_url_key: str = "llm_base_url",
        api_key_key: str = "llm_api_key",
        model_key: str = "llm_model",
        default_model: str = "gpt-4.1-mini",
    ) -> LLMClient:
        api_key = config.get(api_key_key, "")
        base_url = config.get(base_url_key) or DEFAULT_LLM_BASE_URL
        model = config.get(model_key, default_model)
        try:
            timeout = max(1.0, float(config.get("tl_timeout", 30.0)))
        except (TypeError, ValueError):
            timeout = 30.0
        return LLMClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )

    def _translate_qwen_mt(self, texts: list[str], config: dict[str, Any]) -> list[str]:
        if not config.get("qwen_mt_base_url"):
            return ["【未配置Qwen MT URL】"] * len(texts)
        client = self._build_client(
            config,
            base_url_key="qwen_mt_base_url",
            api_key_key="qwen_mt_api_key",
            model_key="qwen_mt_model",
            default_model="qwen-mt-flash",
        )
        api_key = config.get("qwen_mt_api_key", "")
        if client.requires_api_key and not api_key:
            return ["【未配置Key】"] * len(texts)
        options = _build_qwen_mt_options(config)
        results: list[str] = []
        for text in texts:
            try:
                content = client.chat(
                    [{"role": "user", "content": text}],
                    extra_body={"translation_options": options},
                )
            except LLMTimeoutError:
                raise
            except LLMError:
                content = "(失败)"
            results.append(content or "...")
        return results

    def translate(self, texts: list[str], config: dict[str, Any]) -> list[str]:
        if not texts or texts == ["无"]:
            return ["..."] * len(texts)
        if config.get("translation_model_type") == "qwen_mt":
            return self._translate_qwen_mt(texts, config)

        client = self._build_client(config)
        api_key = config.get("llm_api_key", "")
        if client.requires_api_key and not api_key:
            return ["【未配置Key】"] * len(texts)

        use_context = config.get("use_translation_context", True)
        window = config.get("context_window_size", 10)
        context_text = self._build_context_text(use_context, window)
        system_prompt = build_system_prompt(context_text, config)
        text = "\n".join(texts)
        if len(texts) > 1:
            text = "请逐行翻译，保持输入和输出行数一致：\n" + text
        try:
            thinking_type = "enabled" if config.get("llm_thinking_enabled") else "disabled"
            content = client.chat(
                self._build_messages(system_prompt, text, use_context, window),
                temperature=0.3,
                thinking={"type": thinking_type},
            )
        except LLMTimeoutError:
            raise
        except LLMError:
            return ["(失败)"] * len(texts)

        clean = clean_translation_lines(content, len(texts))
        if use_context:
            for i, orig in enumerate(texts):
                self.context_buffer.append(
                    {"orig": orig, "tran": clean[i] if i < len(clean) else ""}
                )
        return clean
