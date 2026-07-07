"""config 模块单测：默认值合并、缺键补齐、原子写。"""

import json
import os

from livetrans.config import DEFAULT_CONFIG, load_config, save_config


def test_load_missing_file_writes_default(tmp_path):
    p = os.path.join(tmp_path, "config.json")
    cfg = load_config(p)
    assert cfg["asr_engine"] == "local_asr"
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
        json.dump(
            {
                "web": {"web_password": "x"},
                "asr": {
                    "asr_engine": "remote_realtime_asr",
                    "dashscope_api_key": "test-key",
                    "remote_asr_model": "paraformer-realtime-v2",
                    "remote_realtime_asr_model": "qwen3-asr-flash-realtime",
                },
                "vad": {"max_speech_duration": 8.0},
            },
            f,
        )
    cfg = load_config(p)
    assert cfg["web_password"] == "x"
    assert cfg["asr_engine"] == "remote_realtime_asr"
    assert cfg["dashscope_api_key"] == "test-key"
    assert cfg["remote_asr_model"] == "paraformer-realtime-v2"
    assert cfg["remote_realtime_asr_model"] == "qwen3-asr-flash-realtime"
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
    assert loaded["web"]["theme_color"] == DEFAULT_CONFIG["theme_color"]
    assert loaded["translation"]["game_hint"] == "测试杂谈"
    assert loaded["translation"]["streamer_type"] == DEFAULT_CONFIG["streamer_type"]
    assert loaded["translation"]["streamer_name"] == DEFAULT_CONFIG["streamer_name"]
    assert loaded["translation"]["translation_model_type"] == DEFAULT_CONFIG["translation_model_type"]
    assert loaded["translation"]["subtitle_send_mode"] == DEFAULT_CONFIG["subtitle_send_mode"]
    assert loaded["translation"]["subtitle_min_interval"] == DEFAULT_CONFIG["subtitle_min_interval"]
    assert loaded["qwen_mt"]["qwen_mt_base_url"] == DEFAULT_CONFIG["qwen_mt_base_url"]
    assert loaded["qwen_mt"]["qwen_mt_api_key"] == DEFAULT_CONFIG["qwen_mt_api_key"]
    assert loaded["qwen_mt"]["qwen_mt_model"] == DEFAULT_CONFIG["qwen_mt_model"]
    assert loaded["qwen_mt"]["qwen_mt_source_lang"] == DEFAULT_CONFIG["qwen_mt_source_lang"]
    assert loaded["qwen_mt"]["qwen_mt_target_lang"] == DEFAULT_CONFIG["qwen_mt_target_lang"]
    assert loaded["qwen_mt"]["qwen_mt_terms_enabled"] is False
    assert loaded["qwen_mt"]["qwen_mt_tm_list_enabled"] is False
    assert loaded["qwen_mt"]["qwen_mt_domains_enabled"] is False
    assert loaded["asr"]["dashscope_api_key"] == DEFAULT_CONFIG["dashscope_api_key"]
    assert loaded["asr"]["remote_asr_model"] == DEFAULT_CONFIG["remote_asr_model"]
    assert loaded["asr"]["remote_realtime_asr_model"] == DEFAULT_CONFIG["remote_realtime_asr_model"]
    assert loaded["vad"]["vad_device"] == "cpu"
    assert loaded["vad"]["filter_games"] == DEFAULT_CONFIG["filter_games"]
    assert "vad" in loaded
