"""实时处理流水线编排：取流 → ffmpeg 帧 → VAD → 本地 ASR → 翻译 → 推送。

worker_loop 跑在后台线程：从直播流拉 PCM 帧喂 Silero VAD，按节流推送语音置信度
（vad_update）供前端进度条；VAD 切出语音段后用本地 ASR 识别，过滤后异步
LLM 翻译并经 Socket.IO 推送（new_message）+ 落盘。重依赖（vad/asr）在此层实例化。
"""

from __future__ import annotations

import os
import threading
import time

from . import audio, filters, storage
from .logging_util import log
from .stream import get_stream_url

__all__ = ["run_worker_loop"]

_VAD_EMIT_INTERVAL = 0.12  # 语音置信度推送节流（秒）


def _build_vad(config: dict):
    """延迟导入并构造 VADProcessor（重依赖 torch）。"""
    from .vad import VADProcessor

    vad = VADProcessor(
        sample_rate=audio.SAMPLE_RATE,
        threshold=float(config.get("vad_threshold", 0.5)),
        min_speech_duration=float(config.get("min_speech_duration", 1.0)),
        max_speech_duration=float(
            config.get("max_speech_duration", config.get("max_record_time", 8))
        ),
        chunk_duration=0.032,
    )
    vad.update_settings(
        {
            "silence_mode": config.get("silence_mode", "auto"),
            "silence_duration": float(config.get("silence_duration", 0.8)),
        }
    )
    return vad


def _build_asr(config: dict, current_dir: str):
    """延迟导入并构造本地 ASR。首启会下载模型。"""
    from .asr import KotobaWhisperEngine, SenseVoiceEngine

    engine_type = config.get("asr_engine", "kotoba_whisper")
    device = config.get("asr_device", "cuda")
    if engine_type == "local_sensevoice":
        engine = SenseVoiceEngine(device=device, hub="ms")
    else:
        model_dir = config.get("asr_model_dir") or os.path.join(
            "models",
            config.get("asr_model_id", "kotoba-tech/kotoba-whisper-v2.2")
            .rstrip("/")
            .split("/")[-1],
        )
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(current_dir, model_dir)
        engine = KotobaWhisperEngine(
            model_id=config.get("asr_model_id", "kotoba-tech/kotoba-whisper-v2.2"),
            model_dir=model_dir,
            device=device,
            language=config.get("asr_language", "ja"),
            beams=int(config.get("asr_num_beams", 3)),
            chunk_length_s=int(config.get("asr_chunk_length_s", 15)),
            batch_size=int(config.get("asr_batch_size", 8)),
            stride_length_s=(
                int(config.get("asr_stride_left_s", 5)),
                int(config.get("asr_stride_right_s", 3)),
            ),
        )
    engine.set_language(config.get("asr_language", "auto"))
    return engine


def _translate_and_emit(
    state, texts: list[str], ts: float, asr_ms: int | None = None
) -> None:
    """后台线程：翻译一批文本并逐条推送 + 落盘。"""
    try:
        tl_start = time.perf_counter()
        trans = state.translator.translate(texts, state.config)
        tl_ms = int(round((time.perf_counter() - tl_start) * 1000))
        save_dir = os.path.join(state.current_dir, "output")
        os.makedirs(save_dir, exist_ok=True)
        for i, orig in enumerate(texts):
            tran = trans[i] if i < len(trans) else ""
            item_ts = ts + i * 0.01
            item = {
                "ts": item_ts,
                "orig": orig,
                "tran": tran,
                "asr_ms": asr_ms,
                "tl_ms": tl_ms,
            }
            with state.history_lock:
                state.history_buffer.append(item)
                if len(state.history_buffer) > 50:
                    state.history_buffer[:] = state.history_buffer[-50:]
            state.socketio.emit("new_message", item)
            storage.auto_save_record(save_dir, orig, tran, item_ts)
    except Exception as e:
        log("Error", f"异步翻译出错: {e}")


def run_worker_loop(state) -> None:
    """后台工作线程主循环。由 server 通过 socketio.start_background_task 启动。"""
    config = state.config
    ffmpeg_exe = os.path.join(state.current_dir, "ffmpeg.exe")

    log("Core", "工作线程启动，准备加载本地 ASR 模型...")
    try:
        asr_engine = _build_asr(config, state.current_dir)
        log("Core", "本地 ASR 加载完成")
    except Exception as e:
        log("Error", f"本地 ASR 加载失败，工作线程退出: {e}")
        return

    vad = _build_vad(config)
    proc_holder: dict = {"proc": None}
    last_emit = [0.0]

    def on_proc(p):
        proc_holder["proc"] = p

    while True:
        if not state.is_running:
            time.sleep(1)
            continue

        if state.reload_event.is_set():
            state.reload_event.clear()
            vad = _build_vad(config)
            asr_engine.set_language(config.get("asr_language", "auto"))
            log("Core", "VAD/ASR 参数已重载")

        if not os.path.exists(ffmpeg_exe):
            log("Error", "未找到 ffmpeg.exe")
            time.sleep(5)
            continue

        stream_url = get_stream_url(config.get("bili_room_url", ""))
        if not stream_url:
            log("Core", "无法获取直播流，等待重试...")
            time.sleep(5)
            continue

        log("Core", "直播流已连接，开始识别...")
        try:
            for frame in audio.stream_frames(
                ffmpeg_exe, stream_url, lambda: state.is_running, on_proc
            ):
                seg = vad.process_chunk(frame)

                now = time.time()
                if now - last_emit[0] >= _VAD_EMIT_INTERVAL:
                    last_emit[0] = now
                    pct = int(round(vad.last_confidence * 100))
                    state.socketio.emit("vad_update", {"confidence": pct})

                if seg is None:
                    continue

                try:
                    asr_start = time.perf_counter()
                    result = asr_engine.transcribe(seg)
                    asr_ms = int(round((time.perf_counter() - asr_start) * 1000))
                except Exception as e:
                    log("Error", f"ASR 识别出错: {e}")
                    continue
                if not result or not result.get("text"):
                    continue

                blacklist = filters.parse_banned_words(config.get("banned_words", ""))
                text = filters.filter_text(result["text"], blacklist)
                if not text:
                    continue

                log("ASR", f"识别: {text}")
                state.socketio.start_background_task(
                    _translate_and_emit, state, [text], time.time(), asr_ms
                )
        except Exception as e:
            log("Error", f"处理循环异常: {e}")
        finally:
            proc_holder["proc"] = None
            log("Core", "音频处理循环退出，等待重连...")
