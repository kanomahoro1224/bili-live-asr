"""config 模块单测：默认值合并、缺键补齐、原子写。"""

import json
import os

from livetrans.config import DEFAULT_CONFIG, load_config, save_config


def test_load_missing_file_writes_default(tmp_path):
    p = os.path.join(tmp_path, "config.json")
    cfg = load_config(p)
    assert cfg["asr_engine"] == "kotoba_whisper"
    assert cfg["asr_model_id"] == "kotoba-tech/kotoba-whisper-v2.2"
    assert cfg["asr_model_dir"] == "models/kotoba-whisper-v2.2"
    assert os.path.exists(p)


def test_load_merges_missing_keys(tmp_path):
    p = os.path.join(tmp_path, "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"web_password": "x"}, f)
    cfg = load_config(p)
    assert cfg["web_password"] == "x"          # 保留用户值
    assert cfg["llm_model"] == DEFAULT_CONFIG["llm_model"]  # 补齐缺键


def test_load_grouped_config_returns_flat_runtime_values(tmp_path):
    p = os.path.join(tmp_path, "config.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"web": {"web_password": "x"}, "vad": {"max_speech_duration": 8.0}}, f)
    cfg = load_config(p)
    assert cfg["web_password"] == "x"
    assert cfg["max_speech_duration"] == 8.0
    assert cfg["max_record_time"] == 8.0


def test_load_corrupt_returns_default(tmp_path):
    p = os.path.join(tmp_path, "config.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write("{ not valid json")
    cfg = load_config(p)
    assert cfg["asr_engine"] == DEFAULT_CONFIG["asr_engine"]


def test_save_is_atomic_and_roundtrips(tmp_path):
    p = os.path.join(tmp_path, "config.json")
    data = DEFAULT_CONFIG.copy()
    data["game_hint"] = "测试杂谈"
    save_config(p, data)
    with open(p, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["translation"]["game_hint"] == "测试杂谈"
    assert "vad" in loaded
