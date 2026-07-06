"""轻量日志工具（纯逻辑，无第三方依赖）。

log() 打印带时间戳的 [tag] 消息并存入内存环形缓冲，Web 的 /logs 接口读取
log_buffer 供前端轮询展示。
"""

from __future__ import annotations

import collections
import time

__all__ = ["log", "log_buffer"]

log_buffer: "collections.deque[str]" = collections.deque(maxlen=200)


def log(tag: str, msg: str) -> None:
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    entry = f"[{timestamp}] [{tag}] {msg}"
    print(entry)
    log_buffer.append(entry)
