"""用 ffmpeg 把直播流解码为 PCM 帧（仅依赖 ffmpeg 子进程 + numpy）。

stream_frames() 启动 ffmpeg 拉流并按固定块大小产出 float32 单声道帧（16kHz），
供 VAD 逐帧处理。调用方通过 should_run() 回调控制何时停止，退出时回收 ffmpeg。
"""

from __future__ import annotations

import queue
import subprocess
import threading
from typing import Callable, Iterator

import numpy as np

__all__ = ["SAMPLE_RATE", "CHUNK_SIZE", "stream_frames"]

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 512 样本 @16k = 32ms，匹配 Silero VAD 原生窗口


def stream_frames(
    ffmpeg_exe: str,
    stream_url: str,
    should_run: Callable[[], bool],
    on_proc: Callable[[subprocess.Popen], None] | None = None,
    yield_idle: bool = False,
    idle_timeout: float = 0.2,
) -> Iterator[np.ndarray | None]:
    """拉流并逐帧 yield float32 音频块（长度 CHUNK_SIZE，范围 [-1,1]）。

    should_run(): 返回 False 时停止读取并结束 ffmpeg。
    on_proc(proc): 可选回调，拿到 Popen 句柄（供外部强制终止）。
    数据不足一帧或流中断时正常结束生成器。
    """
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "quiet", "-i", stream_url,
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=CHUNK_SIZE * 2
    )
    if on_proc is not None:
        on_proc(proc)
    chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=8)

    def _read_stdout() -> None:
        try:
            while proc.poll() is None:
                chunk = proc.stdout.read(CHUNK_SIZE * 2)
                if not chunk:
                    break
                chunks.put(chunk)
        except Exception:
            pass
        finally:
            try:
                chunks.put_nowait(None)
            except queue.Full:
                pass

    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    try:
        while should_run() and proc.poll() is None:
            try:
                chunk = chunks.get(timeout=idle_timeout)
            except queue.Empty:
                if yield_idle:
                    yield None
                continue
            if chunk is None:
                break
            if not chunk or len(chunk) != CHUNK_SIZE * 2:
                break
            audio_i16 = np.frombuffer(chunk, dtype=np.int16)
            yield audio_i16.astype(np.float32) / 32768.0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        reader.join(timeout=0.5)
