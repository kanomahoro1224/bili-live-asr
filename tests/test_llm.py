import pytest

from livetrans import llm
from livetrans.llm import LLMClient, LLMError, LLMTimeoutError


class Resp:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_llm_client_extracts_string_content(monkeypatch):
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: Resp({"choices": [{"message": {"content": "  hi  "}}]}),
    )

    assert LLMClient(base_url="http://localhost:11434/v1").chat([]) == "hi"


def test_llm_client_extracts_array_content(monkeypatch):
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: Resp(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "he"},
                                {"type": "text", "text": "llo"},
                            ]
                        }
                    }
                ]
            }
        ),
    )

    assert LLMClient(base_url="http://localhost:11434/v1").chat([]) == "hello"


def test_llm_client_extracts_legacy_text(monkeypatch):
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: Resp({"choices": [{"text": " SEND "}]}),
    )

    assert LLMClient(base_url="http://localhost:11434/v1").chat([]) == "SEND"


def test_llm_client_does_not_use_reasoning_as_final_content(monkeypatch):
    monkeypatch.setattr(
        llm.requests,
        "post",
        lambda *args, **kwargs: Resp(
            {"choices": [{"message": {"content": "", "reasoning_content": "SEND"}}]}
        ),
    )

    assert LLMClient(base_url="http://localhost:11434/v1").chat([]) == ""


def test_llm_client_raises_on_missing_content(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *args, **kwargs: Resp({"choices": [{}]}))

    with pytest.raises(LLMError):
        LLMClient(base_url="http://localhost:11434/v1").chat([])


def test_llm_client_raises_timeout_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise llm.requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(llm.requests, "post", raise_timeout)

    with pytest.raises(LLMTimeoutError):
        LLMClient(base_url="http://localhost:11434/v1", timeout=1).chat([])
