"""共享运行时状态（纯容器，无重依赖）。

把在线版散落的模块级全局（config / is_running / history_buffer / socketio 等）
收拢到一个 AppState 实例，由 server 创建后注入 web 与 pipeline，避免循环导入与裸全局。
"""

from __future__ import annotations

import threading
from typing import Any

__all__ = ["AppState"]


class AppState:
    def __init__(self, config: dict[str, Any], config_path: str, current_dir: str):
        self.config = config
        self.config_lock = threading.RLock()
        self.config_path = config_path
        self.current_dir = current_dir

        self.is_running = False
        self.history_buffer: list[dict[str, Any]] = []
        self.history_lock = threading.Lock()
        self.subtitle_send_lock = threading.Lock()
        self.last_subtitle_send_at = 0.0

        # 由 server 装配：socketio 实例、翻译器、以及重建 ASR 的回调
        self.socketio = None
        self.translator = None
        # pipeline 用：标记 ASR/VAD 配置已变更需重建
        self.reload_event = threading.Event()
        self.stream_reload_event = threading.Event()
