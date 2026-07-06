import csv
import json
import time

from livetrans.logging_util import log, log_buffer
from livetrans.state import AppState
from livetrans.storage import auto_save_record


def test_auto_save_record_writes_csv_and_json(tmp_path):
    ts = time.mktime((2026, 1, 2, 3, 4, 5, 0, 0, -1))

    auto_save_record(str(tmp_path), "原文", "译文", ts)

    csv_path = tmp_path / "2026-01-02.csv"
    json_path = tmp_path / "2026-01-02.json"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["时间", "原文", "译文"], ["2026-01-02 03:04:05", "原文", "译文"]]
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["orig"] == "原文"
    assert data[0]["tran"] == "译文"


def test_log_appends_to_ring_buffer(capsys):
    before = len(log_buffer)
    log("Test", "hello")
    captured = capsys.readouterr()

    assert "[Test] hello" in captured.out
    assert len(log_buffer) == before + 1
    assert "[Test] hello" in log_buffer[-1]


def test_app_state_initializes_runtime_flags(tmp_path):
    state = AppState({"a": 1}, str(tmp_path / "config.json"), str(tmp_path))

    assert state.config["a"] == 1
    assert not state.is_running
    assert state.history_buffer == []
    assert state.socketio is None
    assert not state.reload_event.is_set()
    assert not state.stream_reload_event.is_set()
