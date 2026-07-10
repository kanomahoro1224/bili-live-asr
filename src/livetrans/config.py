"""配置加载/保存与默认值（纯逻辑，不依赖 torch / 网络 / flask）。

运行时配置以 JSON 落盘（config.json）。load_config 把磁盘值与 DEFAULT_CONFIG
合并、补齐缺键；save_config 原子写盘防崩溃损坏。默认引擎为本地
kotoba_whisper，同时保留远程 Qwen/DashScope ASR 配置。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

__all__ = [
    "DEFAULT_CONFIG",
    "CONFIG_SECTIONS",
    "load_config",
    "save_config",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "web_password": "admin",
    "theme_color": "#4f46e5",
    "streamer_type": "Vtuber",
    "streamer_name": "鹿乃",
    "game_hint": "杂谈",
    "prompt_extra": "",
    "bili_room_url": "https://live.bilibili.com/000000",
    "bili_room_id": "000000",
    "bili_cookie": "",
    "bili_csrf": "",
    # ASR 引擎：KITS-main 同款 Kotoba Whisper
    "asr_engine": "local_asr",
    "asr_model_id": "kotoba-tech/kotoba-whisper-v2.2",
    "asr_model_dir": "models/kotoba-whisper-v2.2",
    "asr_device": "cuda",                 # cuda / cpu
    "asr_language": "auto",               # auto / ja / zh / en / ko
    "asr_num_beams": 3,
    "asr_chunk_length_s": 15,
    "asr_batch_size": 8,
    "asr_stride_left_s": 5,
    "asr_stride_right_s": 3,
    "dashscope_api_key": "",
    "remote_asr_model": "paraformer-realtime-v2",
    "remote_asr_timeout": 8.0,
    "remote_realtime_asr_model": "qwen3-asr-flash-realtime",
    "remote_realtime_asr_url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    "remote_realtime_asr_timeout": 8.0,
    "incremental_asr": False,
    "interim_interval": 2.0,
    # VAD & 过滤参数
    "vad_threshold": 0.5,                 # Silero 触发阈值
    "vad_device": "cpu",
    "min_silence_duration": 0.6,          # 固定静音切分时长（秒）
    "min_speech_duration": 1.0,           # 最短语音段（秒）
    "max_speech_duration": 8.0,           # 最长语音段（秒），超时回溯切分
    "max_record_time": 8.0,               # 旧配置名：兼容 Web 表单
    "silence_mode": "auto",               # auto / fixed
    "silence_duration": 0.8,              # fixed 模式下的静音切分时长（秒）
    "banned_words": "視聴, 字幕, MBC, Music, music, BGM, VIDEO, WATCH, Subscribe",
    "filter_games": "",
    # OpenAI-compatible LLM API
    "llm_base_url": "https://api.openai.com/v1",
    "llm_api_key": "",
    "llm_model": "gpt-4.1-mini",
    "translation_model_type": "llm",
    "llm_thinking_enabled": False,
    "tl_timeout": 30.0,
    "qwen_mt_source_lang": "Japanese",
    "qwen_mt_target_lang": "Chinese",
    "qwen_mt_base_url": "",
    "qwen_mt_api_key": "",
    "qwen_mt_model": "qwen-mt-flash",
    "qwen_mt_terms_enabled": False,
    "qwen_mt_terms": [],
    "qwen_mt_tm_list_enabled": False,
    "qwen_mt_tm_list": [],
    "qwen_mt_domains_enabled": False,
    "qwen_mt_domains": "",
    "subtitle_review_base_url": "",
    "subtitle_review_api_key": "",
    "subtitle_review_model": "",
    "subtitle_review_thinking_enabled": False,
    # 字幕发送模式
    "subtitle_send_mode": "manual",       # manual / auto
    "subtitle_min_interval": 2.0,
    # 安全设置
    "log_security_events": True,
    # 翻译上下文设置
    "use_translation_context": True,
    "context_window_size": 10,
    "max_context_buffer": 20,
}

CONFIG_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("web", ("web_password", "theme_color")),
    ("stream", ("bili_room_url", "bili_room_id")),
    ("bilibili_danmu", ("bili_cookie", "bili_csrf")),
    (
        "asr",
        (
            "asr_engine",
            "asr_model_id",
            "asr_model_dir",
            "asr_device",
            "asr_language",
            "asr_num_beams",
            "asr_chunk_length_s",
            "asr_batch_size",
            "asr_stride_left_s",
            "asr_stride_right_s",
            "dashscope_api_key",
            "remote_asr_model",
            "remote_asr_timeout",
            "remote_realtime_asr_model",
            "remote_realtime_asr_url",
            "remote_realtime_asr_timeout",
            "incremental_asr",
            "interim_interval",
        ),
    ),
    (
        "vad",
        (
            "vad_threshold",
            "vad_device",
            "min_speech_duration",
            "max_speech_duration",
            "silence_mode",
            "silence_duration",
            "banned_words",
            "filter_games",
        ),
    ),
    (
        "translation",
        (
            "game_hint",
            "streamer_type",
            "streamer_name",
            "prompt_extra",
            "llm_base_url",
            "llm_api_key",
            "llm_model",
            "translation_model_type",
            "llm_thinking_enabled",
            "tl_timeout",
            "subtitle_review_base_url",
            "subtitle_review_api_key",
            "subtitle_review_model",
            "subtitle_review_thinking_enabled",
            "subtitle_send_mode",
            "subtitle_min_interval",
            "use_translation_context",
            "context_window_size",
            "max_context_buffer",
        ),
    ),
    (
        "qwen_mt",
        (
            "qwen_mt_base_url",
            "qwen_mt_api_key",
            "qwen_mt_model",
            "qwen_mt_source_lang",
            "qwen_mt_target_lang",
            "qwen_mt_terms_enabled",
            "qwen_mt_terms",
            "qwen_mt_tm_list_enabled",
            "qwen_mt_tm_list",
            "qwen_mt_domains_enabled",
            "qwen_mt_domains",
        ),
    ),
    ("security", ("log_security_events",)),
)

_LEGACY_WRITE_SKIP = {
    "legacy",
    "max_record_time",
    "min_silence_duration",
    "deepseek_key",
    "deepseek_model",
    "remote_asr_url",
    "use_vad",
    "no_speech_threshold",
    "min_avg_logprob",
}


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """支持旧版扁平配置和新版按模块分组配置，返回运行时扁平字典。"""
    flat: dict[str, Any] = {}
    section_names = {name for name, _ in CONFIG_SECTIONS}
    for key, value in data.items():
        if key in section_names and isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    if "max_record_time" in flat and "max_speech_duration" not in flat:
        flat["max_speech_duration"] = flat["max_record_time"]
    if "max_speech_duration" in flat:
        flat["max_record_time"] = flat["max_speech_duration"]
    return flat


def _sectioned_config(config: dict[str, Any]) -> dict[str, Any]:
    """把运行时扁平配置整理成写盘用的模块分组。"""
    flat = _flatten_config(config)
    sectioned: dict[str, Any] = {}
    used: set[str] = set()
    for section, keys in CONFIG_SECTIONS:
        values = {key: flat[key] for key in keys if key in flat}
        if values:
            sectioned[section] = values
            used.update(keys)

    extras = {
        key: value
        for key, value in flat.items()
        if key not in used and key not in _LEGACY_WRITE_SKIP
    }
    if extras:
        sectioned["legacy"] = extras
    return sectioned


def load_config(path: str) -> dict[str, Any]:
    """读取配置文件并与默认值合并；文件不存在则写出默认值并返回。"""
    if not os.path.exists(path):
        cfg = DEFAULT_CONFIG.copy()
        save_config(path, cfg)
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()
    if not isinstance(saved, dict):
        return DEFAULT_CONFIG.copy()
    cfg = _flatten_config(saved)
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    cfg["max_record_time"] = cfg["max_speech_duration"]
    return cfg


def save_config(path: str, config: dict[str, Any]) -> None:
    """原子写盘：先写临时文件再 os.replace，避免崩溃时损坏配置。"""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_sectioned_config(config), f, indent=4, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
