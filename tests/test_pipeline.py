import os
import queue
import threading

import numpy as np
import pytest

from livetrans.state import AppState
from livetrans import pipeline
from livetrans.llm import LLMTimeoutError
from livetrans.pipeline import _is_remote_realtime_asr


def test_remote_realtime_asr_bypasses_vad_mode_detection():
    assert _is_remote_realtime_asr({"asr_engine": "remote_realtime_asr"})
    assert _is_remote_realtime_asr({"asr_engine": "qwen_asr"})
    assert not _is_remote_realtime_asr({"asr_engine": "remote_asr"})
    assert not _is_remote_realtime_asr({"asr_engine": "local_asr"})


def test_enqueue_translation_drops_oldest_when_full():
    q = queue.Queue(maxsize=1)
    q.put_nowait((["old"], 1.0, None))

    pipeline._enqueue_translation(q, ["new"], 2.0, 3)

    assert q.get_nowait() == (["new"], 2.0, 3)


def test_translate_and_emit_saves_history_and_auto_subtitle(monkeypatch, tmp_path):
    class Translator:
        def translate(self, texts, config):
            return [f"{text}-译" for text in texts]

    class Socket:
        def __init__(self):
            self.events = []

        def emit(self, name, data):
            self.events.append((name, data))

    state = AppState(
        {"subtitle_send_mode": "auto", "llm_api_key": "key"},
        str(tmp_path / "config.json"),
        str(tmp_path),
    )
    state.translator = Translator()
    state.socketio = Socket()
    subtitle_queue = queue.Queue()
    saved = []
    monkeypatch.setattr(pipeline.storage, "auto_save_record", lambda *args: saved.append(args))

    pipeline._translate_and_emit(state, ["原文"], 100.0, asr_ms=12, subtitle_queue=subtitle_queue)

    assert state.history_buffer[0]["orig"] == "原文"
    assert state.history_buffer[0]["tran"] == "原文-译"
    assert state.socketio.events[0][0] == "new_message"
    assert saved
    assert subtitle_queue.get_nowait()["tran"] == "原文-译"


def test_translate_and_emit_drops_timeout_task(monkeypatch, tmp_path):
    class Translator:
        def translate(self, texts, config):
            raise LLMTimeoutError("timed out")

    class Socket:
        def __init__(self):
            self.events = []

        def emit(self, name, data):
            self.events.append((name, data))

    state = AppState(
        {"subtitle_send_mode": "auto", "llm_api_key": "key", "tl_timeout": 9},
        str(tmp_path / "config.json"),
        str(tmp_path),
    )
    state.translator = Translator()
    state.socketio = Socket()
    subtitle_queue = queue.Queue()
    saved = []
    logs = []
    monkeypatch.setattr(pipeline.storage, "auto_save_record", lambda *args: saved.append(args))
    monkeypatch.setattr(pipeline, "log", lambda tag, msg: logs.append((tag, msg)))

    pipeline._translate_and_emit(state, ["原文"], 100.0, asr_ms=12, subtitle_queue=subtitle_queue)

    assert not state.history_buffer
    assert state.socketio.events == []
    assert saved == []
    assert subtitle_queue.empty()
    assert logs == [("TL", "翻译超时 9s，丢弃任务：原文")]


def test_handle_asr_result_filters_and_enqueues(monkeypatch, tmp_path):
    state = AppState({"banned_words": "BGM", "filter_games": ""}, str(tmp_path / "c.json"), str(tmp_path))
    q = queue.Queue()
    monkeypatch.setattr(pipeline.time, "time", lambda: 123.0)

    pipeline._handle_asr_result(state, q, {"text": " こんにちは "}, asr_ms=5)

    assert q.get_nowait() == (["こんにちは"], 123.0, 5)
    pipeline._handle_asr_result(state, q, {"text": "BGM"}, asr_ms=5)
    assert q.empty()


def test_split_sentences_uses_punctuation_and_comma_fallback():
    assert pipeline._split_sentences("これはテストです。まだ途中", "ja") == [
        "これはテストです。",
        "まだ途中",
    ]
    long_text = "これはかなり長いテスト文章なので、ここで一度切ってほしいです"
    assert pipeline._split_sentences(long_text, "ja") == [
        "これはかなり長いテスト文章なので、",
        "ここで一度切ってほしいです",
    ]


def test_do_interim_asr_commits_complete_sentence(monkeypatch, tmp_path):
    class Vad:
        def __init__(self):
            self.trimmed = None

        def peek_buffer(self):
            return np.zeros(32000, dtype=np.float32), 2.0

        def trim_front(self, samples):
            self.trimmed = samples

    class Engine:
        def transcribe(self, audio):
            return {
                "text": "これはとても長いテスト文章です。まだ話している",
                "language": "ja",
            }

    state = AppState({}, str(tmp_path / "c.json"), str(tmp_path))
    q = queue.Queue()
    vad = Vad()
    interim = {"pending": "", "committed_tail": "", "active": False}

    assert pipeline._do_interim_asr(state, q, Engine(), vad, interim)

    texts, _, asr_ms = q.get_nowait()
    assert texts == ["これはとても長いテスト文章です。"]
    assert asr_ms >= 0
    assert vad.trimmed is not None
    assert interim["active"] is True
    assert interim["committed_tail"] == "これはとても長いテスト文章です。"


def test_flush_vad_with_silence_transcribes_pending_segment(tmp_path):
    class Vad:
        _is_speaking = True

        def _get_effective_silence_limit(self):
            return 0

        def process_chunk(self, chunk):
            return np.ones(512, dtype=np.float32)

    class Engine:
        def transcribe(self, audio):
            return {"text": "最終セグメント", "language": "ja"}

    state = AppState({}, str(tmp_path / "c.json"), str(tmp_path))
    q = queue.Queue()
    interim = {"active": False}

    pipeline._flush_vad_with_silence(state, q, Engine(), Vad(), interim)

    assert q.get_nowait()[0] == ["最終セグメント"]


def test_build_asr_selects_remote_and_local(monkeypatch, tmp_path):
    calls = []

    class Engine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

        def set_language(self, language):
            self.language = language

    monkeypatch.setattr("livetrans.asr.QwenRealtimeEngine", Engine)
    monkeypatch.setattr("livetrans.asr.DashScopeRemoteEngine", Engine)
    monkeypatch.setattr("livetrans.asr.KotobaWhisperEngine", Engine)

    qwen = pipeline._build_asr(
        {"asr_engine": "remote_realtime_asr", "dashscope_api_key": "k", "asr_language": "ja"},
        str(tmp_path),
    )
    local = pipeline._build_asr(
        {"asr_engine": "local_asr", "asr_model_dir": "models/m", "asr_device": "cpu"},
        str(tmp_path),
    )

    assert qwen.kwargs["api_key"] == "k"
    assert os.path.normpath(local.kwargs["model_dir"]) == os.path.normpath(
        str(tmp_path / "models" / "m")
    )


def test_wait_subtitle_interval_updates_timestamp(monkeypatch, tmp_path):
    state = AppState({}, str(tmp_path / "c.json"), str(tmp_path))
    state.subtitle_send_lock = threading.Lock()
    state.last_subtitle_send_at = 10.0
    slept = []
    times = iter([11.0, 12.0])
    monkeypatch.setattr(pipeline.time, "time", lambda: next(times))
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: slept.append(seconds))

    pipeline._wait_subtitle_interval(state, {"subtitle_min_interval": 2.0})

    assert slept == [1.0]
    assert state.last_subtitle_send_at == 12.0


def test_auto_subtitle_worker_emits_unsent_when_review_skips(monkeypatch, tmp_path):
    class Socket:
        def __init__(self):
            self.events = []

        def emit(self, name, data):
            self.events.append((name, data))

    class OneItemQueue:
        def get(self):
            return {"orig": "原文", "tran": "译文", "ts": 123.0}

        def task_done(self):
            raise RuntimeError("stop")

    state = AppState(
        {"subtitle_send_mode": "auto", "llm_api_key": "key"},
        str(tmp_path / "config.json"),
        str(tmp_path),
    )
    state.socketio = Socket()
    monkeypatch.setattr(pipeline, "_verify_subtitle_with_llm", lambda *args: False)

    with pytest.raises(RuntimeError, match="stop"):
        pipeline._auto_subtitle_worker(state, OneItemQueue())

    assert state.socketio.events == [
        ("subtitle_status", {"ts": 123.0, "status": "unsent"})
    ]


@pytest.mark.parametrize(
    ("send_result", "status"),
    [({"ok": True}, "sent"), ({"ok": False, "error": "bad"}, "failed")],
)
def test_auto_subtitle_worker_emits_send_result_status(
    monkeypatch, tmp_path, send_result, status
):
    class Socket:
        def __init__(self):
            self.events = []

        def emit(self, name, data):
            self.events.append((name, data))

    class OneItemQueue:
        def get(self):
            return {"orig": "原文", "tran": "译文", "ts": 123.0}

        def task_done(self):
            raise RuntimeError("stop")

    state = AppState(
        {"subtitle_send_mode": "auto", "llm_api_key": "key"},
        str(tmp_path / "config.json"),
        str(tmp_path),
    )
    state.socketio = Socket()
    monkeypatch.setattr(pipeline, "_verify_subtitle_with_llm", lambda *args: True)
    monkeypatch.setattr(pipeline, "_wait_subtitle_interval", lambda *args: None)
    monkeypatch.setattr(pipeline.security, "send_danmu", lambda *args: send_result)

    with pytest.raises(RuntimeError, match="stop"):
        pipeline._auto_subtitle_worker(state, OneItemQueue())

    assert state.socketio.events == [
        ("subtitle_status", {"ts": 123.0, "status": status})
    ]


def test_verify_subtitle_logs_model_output_and_parsed_result(monkeypatch):
    logs = []
    calls = []

    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            self.model = model

        def chat(self, messages, **params):
            calls.append(params)
            return "send\n"

    monkeypatch.setattr(pipeline, "LLMClient", Client)
    monkeypatch.setattr(pipeline, "render_prompt", lambda name, **values: "prompt")
    monkeypatch.setattr(pipeline, "log", lambda tag, msg: logs.append((tag, msg)))

    assert pipeline._verify_subtitle_with_llm(
        "译文", "原文", {"llm_model": "review-model", "llm_base_url": "http://local/v1"}
    )
    assert logs == [
        (
            "Subtitle",
            "subtitle_review 模型=review-model 输出='send' 解析结果=SEND",
        )
    ]
    assert calls == [{"temperature": 0, "thinking": {"type": "disabled"}}]


def test_verify_subtitle_logs_empty_output(monkeypatch):
    logs = []

    class Client:
        requires_api_key = False

        def __init__(self, base_url, api_key, model, timeout):
            pass

        def chat(self, messages, **params):
            return ""

    monkeypatch.setattr(pipeline, "LLMClient", Client)
    monkeypatch.setattr(pipeline, "render_prompt", lambda name, **values: "prompt")
    monkeypatch.setattr(pipeline, "log", lambda tag, msg: logs.append((tag, msg)))

    assert not pipeline._verify_subtitle_with_llm(
        "译文", "原文", {"llm_model": "review-model", "llm_base_url": "http://local/v1"}
    )
    assert logs == [
        ("Subtitle", "subtitle_review 空输出：模型=review-model"),
        ("Subtitle", "subtitle_review 模型=review-model 输出='' 解析结果=SKIP"),
    ]
