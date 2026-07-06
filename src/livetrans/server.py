"""组装与启动：缓存环境 → 配置 → 状态 → Web → 后台流水线 → 运行 SocketIO。

关键顺序：在任何 torch/funasr 导入之前调用 model_manager.apply_cache_env()，
把模型缓存指向 tong/models/。run() 启动后台 worker_loop 与 Flask-SocketIO 服务。
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser


def _setup_cache_env(current_dir: str) -> None:
    """在 import torch 之前设置模型缓存目录（指向 tong/models/）。

    model_manager 默认把 MODELS_DIR 取在包目录下，这里改写为 tong/models/ 再 apply。
    """
    from . import model_manager
    from pathlib import Path

    model_manager.MODELS_DIR = Path(current_dir) / "models"
    model_manager.apply_cache_env()


def _find_available_port(preferred: int = 5231) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def run() -> None:
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # current_dir = tong/ （src 的上一级）；__file__ = tong/src/livetrans/server.py
    _setup_cache_env(current_dir)

    from .config import load_config
    from .logging_util import log
    from .state import AppState
    from .translator import OpenAICompatibleTranslator
    from .web import create_web
    from . import pipeline
    from .ffmpeg import require_ffmpeg

    config_path = os.path.join(current_dir, "config.json")
    config = load_config(config_path)

    state = AppState(config=config, config_path=config_path, current_dir=current_dir)
    state.translator = OpenAICompatibleTranslator(
        max_context_buffer=config.get("max_context_buffer", 20)
    )

    app, socketio = create_web(state)

    log("Init", "=== AI 同传系统启动（本地 Kotoba Whisper 版）===")
    try:
        ffmpeg_exe = require_ffmpeg()
    except FileNotFoundError as e:
        log("Error", str(e))
        sys.exit(1)
    log("Init", f"ffmpeg: {ffmpeg_exe}")

    socketio.start_background_task(pipeline.run_worker_loop, state)

    port = _find_available_port(5231)
    if port != 5231:
        log("Init", f"端口 5231 被占用，改用 {port}")
    log("Init", f"服务已启动: http://127.0.0.1:{port}")

    def _open_browser():
        import time
        time.sleep(1.5)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        socketio.run(app, host="0.0.0.0", port=port, debug=False)
    except KeyboardInterrupt:
        log("Info", "用户中断，正在关闭...")
    finally:
        log("Info", "服务已关闭")
