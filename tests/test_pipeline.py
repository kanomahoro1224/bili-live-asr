import os
import queue
import threading

from livetrans.state import AppState
from livetrans import pipeline
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


def test_handle_asr_result_filters_and_enqueues(monkeypatch, tmp_path):
    state = AppState({"banned_words": "BGM", "filter_games": ""}, str(tmp_path / "c.json"), str(tmp_path))
    q = queue.Queue()
    monkeypatch.setattr(pipeline.time, "time", lambda: 123.0)

    pipeline._handle_asr_result(state, q, {"text": " こんにちは "}, asr_ms=5)

    assert q.get_nowait() == (["こんにちは"], 123.0, 5)
    pipeline._handle_asr_result(state, q, {"text": "BGM"}, asr_ms=5)
    assert q.empty()


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
