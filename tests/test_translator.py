"""translator 模块单测：提示词拼装、行号清洗、上下文窗口（不触网）。"""

from livetrans.translator import (
    OpenAICompatibleTranslator,
    build_prompt,
    clean_translation_lines,
)
from livetrans.llm import LLMClient


def test_build_prompt_without_context():
    p = build_prompt(["こんにちは"], "", "杂谈", "")
    assert "待翻译内容：" in p
    assert "前文对话上下文" not in p
    assert "こんにちは" in p


def test_build_prompt_with_context():
    p = build_prompt(["A"], "原文: X\n译文: Y", "APEX", "保持语气")
    assert "前文对话上下文" in p
    assert "保持语气" in p
    assert "[APEX]" in p


def test_clean_translation_lines_strips_numbering_and_pads():
    out = clean_translation_lines("1. 你好\n2、早上好", expected=3)
    assert out[0] == "你好"
    assert out[1] == "早上好"
    assert out[2] == "..."        # 补齐到 expected


def test_translate_short_circuits():
    t = OpenAICompatibleTranslator()
    assert t.translate([], {}) == []
    assert t.translate(["无"], {}) == ["..."]
    assert t.translate(["x"], {"llm_api_key": ""}) == ["【未配置Key】"]


def test_llm_client_normalizes_base_url():
    assert LLMClient(base_url="http://localhost:11434/v1").chat_url == (
        "http://localhost:11434/v1/chat/completions"
    )
    assert LLMClient(base_url="https://api.example.com/v1/chat/completions").chat_url == (
        "https://api.example.com/v1/chat/completions"
    )


def test_custom_llm_endpoint_can_omit_api_key():
    assert not LLMClient(base_url="http://localhost:11434/v1", api_key="").requires_api_key


def test_context_window_trims_to_size():
    t = OpenAICompatibleTranslator(max_context_buffer=20)
    for i in range(10):
        t.context_buffer.append({"orig": f"o{i}", "tran": f"t{i}"})
    text = t._build_context_text(use_context=True, window=3)
    # 只取最近 3 句 → 6 行（每句原文+译文）
    assert text.count("\n") == 5
    assert "o9" in text and "o7" in text and "o6" not in text
