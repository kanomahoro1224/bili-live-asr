import collections

import numpy as np
import pytest

from livetrans.vad import VADProcessor


def make_vad() -> VADProcessor:
    vad = VADProcessor.__new__(VADProcessor)
    vad.sample_rate = 16000
    vad.device = "cpu"
    vad.threshold = 0.5
    vad.energy_threshold = 0.02
    vad.min_speech_samples = 512
    vad.max_speech_samples = 4096
    vad._chunk_duration = 0.032
    vad.mode = "energy"
    vad._speech_buffer = []
    vad._confidence_history = []
    vad._speech_samples = 0
    vad._is_speaking = False
    vad._silence_counter = 0
    vad._was_trimmed = False
    vad._pre_buffer = collections.deque(maxlen=3)
    vad._silence_mode = "auto"
    vad._fixed_silence_dur = 0.8
    vad._silence_limit = 2
    vad._progressive_tiers = [(3.0, 1.0), (6.0, 0.5), (10.0, 0.25)]
    vad._pause_history = collections.deque(maxlen=50)
    vad._adaptive_min = 0.3
    vad._adaptive_max = 2.0
    vad.last_confidence = 0.0
    return vad


def test_update_settings_and_energy_confidence():
    vad = make_vad()
    vad.update_settings(
        {
            "vad_mode": "energy",
            "vad_threshold": 0.4,
            "energy_threshold": 0.1,
            "min_speech_duration": 0.5,
            "max_speech_duration": 2.0,
            "silence_mode": "fixed",
            "silence_duration": 0.64,
        }
    )

    assert vad.threshold == 0.4
    assert vad.min_speech_samples == 8000
    assert vad.max_speech_samples == 32000
    assert vad._silence_limit == 20
    assert vad._energy_confidence(np.ones(512, dtype=np.float32) * 0.1) == pytest.approx(0.5)


def test_process_chunk_flushes_after_silence():
    vad = make_vad()
    vad.min_speech_samples = 512
    vad._silence_limit = 1
    loud = np.ones(512, dtype=np.float32) * 0.1
    quiet = np.zeros(512, dtype=np.float32)

    assert vad.process_chunk(loud) is None
    segment = vad.process_chunk(quiet)

    assert segment is not None
    assert len(segment) == 1024
    assert not vad._is_speaking


def test_peek_trim_and_force_flush():
    vad = make_vad()
    vad._is_speaking = True
    vad._speech_buffer = [
        np.arange(4, dtype=np.float32),
        np.arange(4, 8, dtype=np.float32),
    ]
    vad._confidence_history = [0.9, 0.8]
    vad._speech_samples = 8

    audio, duration = vad.peek_buffer()
    assert audio.tolist() == list(range(8))
    assert duration == 8 / vad.sample_rate

    vad.trim_front(5)
    assert vad._speech_samples == 3
    flushed = vad.force_flush()
    assert flushed.tolist() == [5.0, 6.0, 7.0]
    assert vad.peek_buffer() is None


def test_find_best_split_index_detects_valley():
    vad = make_vad()
    vad._confidence_history = [0.9, 0.9, 0.8, 0.1, 0.1, 0.8, 0.9]

    assert vad._find_best_split_index() > 0
