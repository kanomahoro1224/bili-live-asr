"""直播流地址解析（仅依赖 streamlink）。

从 Bilibili 直播间 URL 解析出可拉取的音视频流地址，交给 audio 层用 ffmpeg 拉取。
"""

from __future__ import annotations

from streamlink import Streamlink

__all__ = ["get_stream_url"]


def get_stream_url(room_url: str) -> str:
    """解析直播间 URL，返回最佳流地址；失败返回空字符串。"""
    if not room_url or "你的房间号" in room_url or "000000" in room_url:
        return ""
    try:
        session = Streamlink()
        streams = session.streams(room_url)
        if not streams:
            return ""
        stream = streams.get("best") or next(iter(streams.values()))
        return stream.to_url()
    except Exception:
        return ""
