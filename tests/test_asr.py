import base64
import queue
import threading
import wave
from http import HTTPStatus

import numpy as np
import pytest

from livetrans.asr import (
    DashScopeRemoteEngine,
    KotobaWhisperEngine,
    QwenRealtimeEngine,
    SAMPLE_RATE,
    SenseVoiceEngine,
)


def test_kotoba_helpers_normalize_language_and_device(tmp_path):
    model_dir = KotobaWhisperEngine._resolve_model_dir("org/model-name", str(tmp_path / "m"))

    assert model_dir == (tmp_path / "m").resolve()
    assert KotobaWhisperEngine._normalize_device("cpu") == "cpu"
    assert KotobaWhisperEngine._normalize_language("ja") == "japanese"
    assert KotobaWhisperEngine._normalize_language("unknown") == "unknown"


def test_kotoba_transcribe_uses_pipeline_result():
    engine = KotobaWhisperEngine.__new__(KotobaWhisperEngine)
    engine.language = "japanese"
    engine.beams = 2
    calls = {}

    def fake_pipe(audio_input, return_timestamps, generate_kwargs):
        calls["audio_input"] = audio_input
        calls["generate_kwargs"] = generate_kwargs
        return {"text": "  hello   world  "}

    engine._pipe = fake_pipe

    result = engine.transcribe(np.ones(10, dtype=np.float32))

    assert result["text"] == "hello world"
    assert calls["audio_input"]["sampling_rate"] == SAMPLE_RATE
    assert calls["generate_kwargs"]["num_beams"] == 2
    assert engine.transcribe(np.array([], dtype=np.float32)) is None


def test_sensevoice_audio_padding_and_text_cleanup(monkeypatch):
    engine = SenseVoiceEngine.__new__(SenseVoiceEngine)
    engine._set_input_padding(0.5, log_change=False)
    audio = np.ones(100, dtype=np.float32)
    padded = engine._prepare_audio_input(audio)

    assert padded.shape[0] == int(SAMPLE_RATE * 0.5)

    class FakeModel:
        def generate(self, **kwargs):
            return [{"text": "<|ja|><|HAPPY|> こんにちは "}]

    engine._model = FakeModel()
    engine.language = None
    engine._use_fp16 = False
    result = engine.transcribe(audio)
    assert result == {"text": "こんにちは", "language": "ja", "language_name": "ja"}


def test_qwen_realtime_drain_and_encode():
    engine = QwenRealtimeEngine.__new__(QwenRealtimeEngine)
    engine.language = "ja"
    engine._results = queue.Queue()
    engine._results.put("  a  b ")
    engine._results.put(" ")

    assert engine._drain_results() == ["a b"]

    encoded = engine._encode_audio(np.array([-2.0, 0.0, 2.0], dtype=np.float32), b"x")
    raw = base64.b64decode(encoded)
    assert raw.endswith(b"x")
    assert len(raw) == 3 * 2 + 1


def test_qwen_transcribe_stream_frame_appends_audio():
    class Conversation:
        def __init__(self):
            self.audio = []

        def append_audio(self, data):
            self.audio.append(data)

    engine = QwenRealtimeEngine.__new__(QwenRealtimeEngine)
    engine.language = "ja"
    engine._lock = threading.Lock()
    engine._conversation = Conversation()
    engine._results = queue.Queue()
    engine._results.put(" テスト ")

    result = engine.transcribe_stream_frame(np.ones(4, dtype=np.float32))

    assert result == [{"text": "テスト", "language": "ja", "language_name": "ja"}]
    assert engine._conversation.audio


def test_dashscope_remote_writes_wav_and_transcribes():
    class Recognition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def call(self, path):
            assert path.endswith(".wav")
            with wave.open(path, "rb") as f:
                assert f.getframerate() == SAMPLE_RATE
            return self

        @property
        def status_code(self):
            return HTTPStatus.OK

        def get_sentence(self):
            return [{"text": " hello "}, {"text": "world"}]

    engine = DashScopeRemoteEngine.__new__(DashScopeRemoteEngine)
    engine.language = "ja"
    engine.model = "model"
    engine._Recognition = Recognition

    assert engine.transcribe(np.ones(8, dtype=np.float32))["text"] == "hello world"
    assert engine.transcribe(np.array([], dtype=np.float32)) is None
