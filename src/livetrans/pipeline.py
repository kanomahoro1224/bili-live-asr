"""实时处理流水线编排：取流 → ffmpeg 帧 → VAD → ASR → 翻译 → 推送。

worker_loop 跑在后台线程：从直播流拉 PCM 帧喂 Silero VAD，按节流推送语音置信度
（vad_update）供前端进度条；VAD 切出语音段后用当前 ASR 引擎识别，过滤后异步
LLM 翻译并经 Socket.IO 推送（new_message）+ 落盘。重依赖（vad/asr）在此层实例化。
"""

from __future__ import annotations

import os
import queue
import threading
import time

from . import audio, filters, storage
from . import security
from .llm import DEFAULT_LLM_BASE_URL, LLMClient, LLMError
from .logging_util import log
from .prompt_loader import render_prompt
from .stream import get_stream_url

__all__ = ["run_worker_loop"]

_TRANSLATION_QUEUE_SIZE = 16
_SUBTITLE_QUEUE_SIZE = 32


def _config_snapshot(state) -> dict:
    with state.config_lock:
        return dict(state.config)

_VAD_EMIT_INTERVAL = 0.12  # 语音置信度推送节流（秒）


def _build_vad(config: dict):
    """延迟导入并构造 VADProcessor（重依赖 torch）。"""
    from .vad import VADProcessor

    vad = VADProcessor(
        sample_rate=audio.SAMPLE_RATE,
        device=config.get("vad_device", "cpu"),
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


def _is_remote_realtime_asr(config: dict) -> bool:
    return config.get("asr_engine") in ("remote_realtime_asr", "qwen_asr")


def _build_asr(config: dict, current_dir: str):
    """延迟导入并构造 ASR。首启本地模式会下载模型。"""
    from .asr import (
        DashScopeRemoteEngine,
        KotobaWhisperEngine,
        QwenRealtimeEngine,
        SenseVoiceEngine,
    )

    engine_type = config.get("asr_engine", "local_asr")
    device = config.get("asr_device", "cuda")
    if engine_type in ("remote_realtime_asr", "qwen_asr"):
        engine = QwenRealtimeEngine(
            api_key=config.get("dashscope_api_key", ""),
            language=config.get("asr_language", "ja"),
            model=config.get(
                "remote_realtime_asr_model",
                config.get("remote_asr_model", "qwen3-asr-flash-realtime"),
            ),
            url=config.get(
                "remote_realtime_asr_url",
                config.get(
                    "remote_asr_url",
                    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
                ),
            ),
            timeout=float(
                config.get(
                    "remote_realtime_asr_timeout",
                    config.get("remote_asr_timeout", 8.0),
                )
            ),
        )
    elif engine_type == "remote_asr":
        engine = DashScopeRemoteEngine(
            api_key=config.get("dashscope_api_key", ""),
            language=config.get("asr_language", "ja"),
            model=config.get("remote_asr_model", "paraformer-realtime-v2"),
            timeout=float(config.get("remote_asr_timeout", 8.0)),
        )
    elif engine_type == "local_sensevoice":
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


def _unload_asr(asr_engine) -> None:
    unload = getattr(asr_engine, "unload", None)
    if not callable(unload):
        return
    try:
        unload()
    except Exception as e:
        log("Core", f"ASR unload failed: {e}")


def _translate_and_emit(
    state,
    texts: list[str],
    ts: float,
    asr_ms: int | None = None,
    subtitle_queue: queue.Queue | None = None,
) -> None:
    """后台线程：翻译一批文本并逐条推送 + 落盘。"""
    try:
        config = _config_snapshot(state)
        tl_start = time.perf_counter()
        trans = state.translator.translate(texts, config)
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
            _enqueue_auto_subtitle(state, subtitle_queue, orig, tran, item_ts)
    except Exception as e:
        log("Error", f"异步翻译出错: {e}")


def _translation_worker(
    state, work_queue: queue.Queue, subtitle_queue: queue.Queue | None
) -> None:
    while True:
        texts, ts, asr_ms = work_queue.get()
        try:
            _translate_and_emit(state, texts, ts, asr_ms, subtitle_queue)
        finally:
            work_queue.task_done()


def _enqueue_translation(
    work_queue: queue.Queue, texts: list[str], ts: float, asr_ms: int | None
) -> None:
    item = (texts, ts, asr_ms)
    try:
        work_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        work_queue.get_nowait()
        work_queue.task_done()
        log("TL", "翻译队列已满，丢弃最旧任务")
    except queue.Empty:
        pass

    try:
        work_queue.put_nowait(item)
    except queue.Full:
        log("TL", "翻译队列仍然满，丢弃当前任务")


def _enqueue_auto_subtitle(
    state, subtitle_queue: queue.Queue | None, orig: str, tran: str, ts: float
) -> None:
    if subtitle_queue is None:
        return
    config = _config_snapshot(state)
    if config.get("subtitle_send_mode", "manual") != "auto":
        return
    text = str(tran or "").strip()
    if not text:
        return
    item = {"orig": orig, "tran": text, "ts": ts}
    try:
        subtitle_queue.put_nowait(item)
        return
    except queue.Full:
        pass

    try:
        subtitle_queue.get_nowait()
        subtitle_queue.task_done()
        log("Subtitle", "自动发送队列已满，丢弃最旧候选")
    except queue.Empty:
        pass

    try:
        subtitle_queue.put_nowait(item)
    except queue.Full:
        log("Subtitle", "自动发送队列仍然满，丢弃当前候选")


def _verify_subtitle_with_llm(text: str, orig: str, config: dict) -> bool:
    api_key = config.get("llm_api_key", "")
    base_url = config.get("llm_base_url") or DEFAULT_LLM_BASE_URL
    model = config.get("llm_model", "gpt-4.1-mini")
    client = LLMClient(base_url=base_url, api_key=api_key, model=model, timeout=20)
    if client.requires_api_key and not api_key:
        log("Subtitle", "自动发送跳过：未配置 LLM API Key")
        return False

    prompt = render_prompt("subtitle_review.txt", orig=orig, text=text)
    try:
        decision = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        ).strip().upper()
    except LLMError as e:
        log("Subtitle", f"自动发送审核失败: {e}")
        return False
    return decision.startswith("SEND")


def _wait_subtitle_interval(state, config: dict) -> None:
    interval = max(2.0, float(config.get("subtitle_min_interval", 2.0)))
    with state.subtitle_send_lock:
        now = time.time()
        wait = interval - (now - state.last_subtitle_send_at)
        if wait > 0:
            time.sleep(wait)
        state.last_subtitle_send_at = time.time()


def _auto_subtitle_worker(state, subtitle_queue: queue.Queue) -> None:
    cookie_file = os.path.join(state.current_dir, "bilicookie.json")
    while True:
        item = subtitle_queue.get()
        try:
            config = _config_snapshot(state)
            if config.get("subtitle_send_mode", "manual") != "auto":
                continue
            text = str(item.get("tran") or "").strip()
            orig = str(item.get("orig") or "").strip()
            if len(text) > 198:
                log("Subtitle", "自动发送跳过：字幕过长")
                continue
            if not _verify_subtitle_with_llm(text, orig, config):
                log("Subtitle", f"自动发送审核未通过: {text}")
                continue
            _wait_subtitle_interval(state, config)
            send_config = _config_snapshot(state)
            if send_config.get("subtitle_send_mode", "manual") != "auto":
                continue
            result = security.send_danmu(f"[{text}]", send_config, cookie_file)
            if result.get("ok"):
                log("Subtitle", f"自动发送: {text}")
            else:
                log("Subtitle", f"自动发送失败: {result.get('error')}")
        finally:
            subtitle_queue.task_done()


def _handle_asr_result(state, translation_queue: queue.Queue, result: dict, asr_ms: int):
    if not result or not result.get("text"):
        return
    current_config = _config_snapshot(state)
    blacklist = filters.parse_banned_words(current_config.get("banned_words", ""))
    try:
        game_callouts = filters.resolve_games(
            filters.parse_filter_games(current_config.get("filter_games", ""))
        )
    except ValueError as e:
        log("Filter", str(e))
        game_callouts = frozenset()
    text = filters.filter_text(result["text"], blacklist, game_callouts)
    if not text:
        return
    log("ASR", f"识别: {text}")
    _enqueue_translation(translation_queue, [text], time.time(), asr_ms)


def run_worker_loop(state) -> None:
    """后台工作线程主循环。由 server 通过 socketio.start_background_task 启动。"""
    config = _config_snapshot(state)
    ffmpeg_exe = os.path.join(state.current_dir, "ffmpeg.exe")

    log("Core", "工作线程启动，准备加载 ASR 引擎...")
    try:
        asr_engine = _build_asr(config, state.current_dir)
        log("Core", "ASR 引擎加载完成")
    except Exception as e:
        log("Error", f"ASR 引擎加载失败，工作线程退出: {e}")
        return

    vad = _build_vad(config)
    proc_holder: dict = {"proc": None}
    last_emit = [0.0]
    translation_queue: queue.Queue = queue.Queue(maxsize=_TRANSLATION_QUEUE_SIZE)
    subtitle_queue: queue.Queue = queue.Queue(maxsize=_SUBTITLE_QUEUE_SIZE)
    if state.socketio is not None:
        state.socketio.start_background_task(
            _translation_worker, state, translation_queue, subtitle_queue
        )
        state.socketio.start_background_task(
            _auto_subtitle_worker, state, subtitle_queue
        )
    else:
        threading.Thread(
            target=_translation_worker,
            args=(state, translation_queue, subtitle_queue),
            daemon=True,
        ).start()
        threading.Thread(
            target=_auto_subtitle_worker,
            args=(state, subtitle_queue),
            daemon=True,
        ).start()

    def on_proc(p):
        proc_holder["proc"] = p

    while True:
        if not state.is_running:
            time.sleep(1)
            continue

        if state.reload_event.is_set():
            state.reload_event.clear()
            config = _config_snapshot(state)
            vad = _build_vad(config)
            _unload_asr(asr_engine)
            try:
                asr_engine = _build_asr(config, state.current_dir)
            except Exception as e:
                log("Error", f"ASR 重载失败: {e}")
                time.sleep(3)
                continue
            log("Core", "VAD/ASR 参数已重载")

        if state.stream_reload_event.is_set():
            state.stream_reload_event.clear()
            config = _config_snapshot(state)
            log("Core", "直播流配置已重载")

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
                ffmpeg_exe,
                stream_url,
                lambda: (
                    state.is_running
                    and not state.reload_event.is_set()
                    and not state.stream_reload_event.is_set()
                ),
                on_proc,
            ):
                if _is_remote_realtime_asr(config):
                    try:
                        asr_start = time.perf_counter()
                        results = asr_engine.transcribe_stream_frame(frame)
                        asr_ms = int(round((time.perf_counter() - asr_start) * 1000))
                    except Exception as e:
                        log("Error", f"远程实时 ASR 识别出错: {e}")
                        continue
                    for result in results:
                        _handle_asr_result(state, translation_queue, result, asr_ms)
                    continue

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

                _handle_asr_result(state, translation_queue, result, asr_ms)
        except Exception as e:
            log("Error", f"处理循环异常: {e}")
        finally:
            proc_holder["proc"] = None
            log("Core", "音频处理循环退出，等待重连...")
