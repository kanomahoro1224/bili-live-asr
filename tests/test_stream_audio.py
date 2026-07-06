import time

import numpy as np

from livetrans import audio, stream


def test_get_stream_url_uses_best_stream(monkeypatch):
    class FakeStream:
        def __init__(self, url):
            self.url = url

        def to_url(self):
            return self.url

    class FakeSession:
        def __init__(self):
            self.options = {}

        def set_option(self, key, value):
            self.options[key] = value

        def streams(self, room_url):
            assert room_url == "https://live.bilibili.com/123"
            return {"best": FakeStream("http://stream")}

    monkeypatch.setattr(stream, "Streamlink", FakeSession)

    assert stream.get_stream_url("https://live.bilibili.com/123") == "http://stream"
    assert stream.get_stream_url("https://live.bilibili.com/000000") == ""


def test_get_stream_url_returns_empty_on_error(monkeypatch):
    class FakeSession:
        def set_option(self, key, value):
            pass

        def streams(self, room_url):
            raise RuntimeError("failed")

    monkeypatch.setattr(stream, "Streamlink", FakeSession)

    assert stream.get_stream_url("https://live.bilibili.com/123") == ""


def test_stream_frames_reads_pcm_and_terminates(monkeypatch):
    samples = np.arange(audio.CHUNK_SIZE, dtype=np.int16)

    class FakeStdout:
        def __init__(self):
            self.calls = 0

        def read(self, size):
            self.calls += 1
            if self.calls == 1:
                return samples.tobytes()
            return b""

    class FakeProc:
        def __init__(self, cmd, stdout, stderr, bufsize):
            self.cmd = cmd
            self.stdout = FakeStdout()
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.killed = True

    seen = {}

    def fake_popen(cmd, stdout, stderr, bufsize):
        proc = FakeProc(cmd, stdout, stderr, bufsize)
        seen["proc"] = proc
        return proc

    monkeypatch.setattr(audio.subprocess, "Popen", fake_popen)

    frames = list(
        audio.stream_frames(
            "ffmpeg",
            "http://stream",
            lambda: True,
            on_proc=lambda proc: seen.setdefault("on_proc", proc),
        )
    )

    assert len(frames) == 1
    assert np.allclose(frames[0], samples.astype(np.float32) / 32768.0)
    assert seen["proc"].cmd[:5] == ["ffmpeg", "-y", "-loglevel", "quiet", "-i"]
    assert seen["proc"].terminated
    assert seen["on_proc"] is seen["proc"]
