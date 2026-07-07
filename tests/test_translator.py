"""translator 模块单测：提示词拼装、行号清洗、上下文窗口（不触网）。"""

from livetrans.translator import (
    OpenAICompatibleTranslator,
    build_prompt,
    clean_translation_lines,
)
from livetrans.llm import LLMClient, LLMTimeoutError


def test_build_prompt_without_context():
    p = build_prompt(["こんにちは"], "", "杂谈", "")
    assert "待翻译内容" in p
    assert "前文对话上下文" not in p
    assert "こんにちは" in p


def test_build_prompt_with_context():
    p = build_prompt(["A"], "原文: X\n译文: Y", "APEX", "保持语气")
    assert "前文对话上下文" in p
    assert "保持语气" in p
    assert "[APEX]" in p


def test_build_prompt_uses_streamer_variables():
    p = build_prompt(["A"], "", "APEX", "", "歌手", "鹿乃")
    assert "歌手" in p
    assert "鹿乃" in p
    assert "Vtuber（鹿乃）" not in p


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


def test_translate_sends_thinking_mode(monkeypatch):
    calls = []

    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            calls.append({"timeout": timeout})

        def chat(self, messages, **params):
            calls.append(params)
            return "你好"

    monkeypatch.setattr("livetrans.translator.LLMClient", Client)
    t = OpenAICompatibleTranslator()

    assert t.translate(["こんにちは"], {"llm_thinking_enabled": False}) == ["你好"]
    assert calls[0]["timeout"] == 30.0
    assert calls[1]["thinking"] == {"type": "disabled"}

    assert t.translate(["こんにちは"], {"llm_thinking_enabled": True}) == ["你好"]
    assert calls[2]["timeout"] == 30.0
    assert calls[3]["thinking"] == {"type": "enabled"}


def test_translate_uses_configured_timeout_and_reraises_timeout(monkeypatch):
    calls = []

    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            calls.append(timeout)

        def chat(self, messages, **params):
            raise LLMTimeoutError("timed out")

    monkeypatch.setattr("livetrans.translator.LLMClient", Client)
    t = OpenAICompatibleTranslator()

    try:
        t.translate(["こんにちは"], {"tl_timeout": 12})
    except LLMTimeoutError:
        pass
    else:
        raise AssertionError("expected timeout")

    assert calls == [12.0]


def test_qwen_mt_uses_translation_options(monkeypatch):
    calls = []

    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            calls.append(
                {
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                    "timeout": timeout,
                }
            )

        def chat(self, messages, **params):
            calls.append({"messages": messages, "params": params})
            return "你好"

    monkeypatch.setattr("livetrans.translator.LLMClient", Client)
    t = OpenAICompatibleTranslator()

    out = t.translate(
        ["こんにちは"],
        {
            "translation_model_type": "qwen_mt",
            "llm_base_url": "https://llm.example/v1",
            "llm_api_key": "llm-key",
            "llm_model": "gpt-4.1-mini",
            "qwen_mt_base_url": "https://qwen.example/v1",
            "qwen_mt_api_key": "qwen-key",
            "qwen_mt_model": "qwen-mt-flash",
            "tl_timeout": 8,
            "qwen_mt_source_lang": "Japanese",
            "qwen_mt_target_lang": "Chinese",
            "qwen_mt_terms_enabled": True,
            "qwen_mt_terms": '[{"source":"鹿乃","target":"Kano"}]',
            "qwen_mt_tm_list_enabled": True,
            "qwen_mt_tm_list": '[{"source":"おはよう","target":"早上好"}]',
            "qwen_mt_domains_enabled": True,
            "qwen_mt_domains": "Translate into a casual livestream subtitle style.",
        },
    )

    assert out == ["你好"]
    assert calls[0] == {
        "base_url": "https://qwen.example/v1",
        "api_key": "qwen-key",
        "model": "qwen-mt-flash",
        "timeout": 8.0,
    }
    assert calls[1]["messages"] == [{"role": "user", "content": "こんにちは"}]
    assert calls[1]["params"] == {
        "extra_body": {
            "translation_options": {
                "source_lang": "Japanese",
                "target_lang": "Chinese",
                "terms": [{"source": "鹿乃", "target": "Kano"}],
                "tm_list": [{"source": "おはよう", "target": "早上好"}],
                "domains": "Translate into a casual livestream subtitle style.",
            }
        }
    }


def test_qwen_mt_requires_separate_base_url(monkeypatch):
    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            raise AssertionError("client should not be built without qwen_mt_base_url")

    monkeypatch.setattr("livetrans.translator.LLMClient", Client)
    t = OpenAICompatibleTranslator()

    assert t.translate(["こんにちは"], {"translation_model_type": "qwen_mt"}) == [
        "【未配置Qwen MT URL】"
    ]


def test_qwen_mt_omits_disabled_enhancements(monkeypatch):
    calls = []

    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            pass

        def chat(self, messages, **params):
            calls.append(params)
            return "你好"

    monkeypatch.setattr("livetrans.translator.LLMClient", Client)
    t = OpenAICompatibleTranslator()

    assert t.translate(
        ["こんにちは"],
        {
            "translation_model_type": "qwen_mt",
            "qwen_mt_base_url": "https://qwen.example/v1",
            "qwen_mt_api_key": "qwen-key",
            "qwen_mt_terms": [{"source": "鹿乃", "target": "Kano"}],
            "qwen_mt_tm_list": [{"source": "おはよう", "target": "早上好"}],
            "qwen_mt_domains": "Translate into a casual livestream subtitle style.",
        },
    ) == ["你好"]
    assert calls[0]["extra_body"]["translation_options"] == {
        "source_lang": "Japanese",
        "target_lang": "Chinese",
    }


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
