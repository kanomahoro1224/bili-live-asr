from livetrans.pipeline import _is_remote_realtime_asr


def test_remote_realtime_asr_bypasses_vad_mode_detection():
    assert _is_remote_realtime_asr({"asr_engine": "remote_realtime_asr"})
    assert _is_remote_realtime_asr({"asr_engine": "qwen_asr"})
    assert not _is_remote_realtime_asr({"asr_engine": "remote_asr"})
    assert not _is_remote_realtime_asr({"asr_engine": "local_asr"})
