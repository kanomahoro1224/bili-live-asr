import os
from pathlib import Path

from livetrans.ffmpeg import find_executable, require_ffmpeg


def test_find_executable_uses_path_not_current_directory(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    tools = tmp_path / "tools"
    cwd.mkdir()
    tools.mkdir()
    suffix = ".exe" if os.name == "nt" else ""
    (cwd / f"ffmpeg{suffix}").write_text("", encoding="utf-8")
    expected = tools / f"ffmpeg{suffix}"
    expected.write_text("", encoding="utf-8")

    monkeypatch.chdir(cwd)
    monkeypatch.setenv("PATH", str(tools))
    if os.name == "nt":
        monkeypatch.setenv("PATHEXT", ".EXE")

    assert Path(find_executable("ffmpeg")).resolve() == expected.resolve()


def test_require_ffmpeg_errors_when_missing(monkeypatch):
    monkeypatch.setenv("PATH", "")
    try:
        require_ffmpeg()
    except FileNotFoundError as exc:
        assert "PATH" in str(exc)
    else:
        raise AssertionError("require_ffmpeg should fail when ffmpeg is missing")
