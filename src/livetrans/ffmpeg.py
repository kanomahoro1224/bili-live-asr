"""Resolve external ffmpeg tools from PATH."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["find_executable", "require_ffmpeg"]


def _candidate_names(command: str) -> list[str]:
    path = Path(command)
    if path.suffix:
        return [command]
    if os.name != "nt":
        return [command]
    suffixes = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
    return [command, *(f"{command}{suffix.lower()}" for suffix in suffixes)]


def find_executable(command: str) -> str | None:
    """Find an executable by scanning PATH only, excluding the current directory."""
    cwd = Path.cwd().resolve()
    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_dir:
            continue
        try:
            directory = Path(raw_dir).resolve()
        except OSError:
            continue
        if directory == cwd:
            continue
        for name in _candidate_names(command):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def require_ffmpeg() -> str:
    ffmpeg = find_executable("ffmpeg")
    if ffmpeg:
        return ffmpeg
    raise FileNotFoundError("未找到 ffmpeg。请安装 ffmpeg，并确认 ffmpeg 在 PATH 中。")
