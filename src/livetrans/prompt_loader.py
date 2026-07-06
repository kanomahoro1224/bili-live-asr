"""Load editable prompt templates from src/livetrans/prompts."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).with_name("prompts")


class _PromptValues(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **values: object) -> str:
    template = load_prompt(name)
    normalized = _PromptValues({key: str(value) for key, value in values.items()})
    return template.format_map(normalized).strip()
