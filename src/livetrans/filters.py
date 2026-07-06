"""Text cleanup and configurable ASR filters."""

from __future__ import annotations

__all__ = [
    "clean_unicode",
    "filter_text",
    "is_game_callout",
    "parse_banned_words",
    "parse_filter_games",
    "resolve_games",
    "supported_games",
]

_EDGE_PUNCT = "。.,，、!！?？…・「」『』【】（）()[]{}　 \t\n\r"


def _normalize(text: str) -> str:
    text = (text or "").replace(" ", "").replace("　", "")
    return text.strip(_EDGE_PUNCT).lower()


_RAW_VALORANT_CALLOUTS: tuple[str, ...] = (
    "アラームボット配置",
    "アラームボット展開",
    "グレネード配置",
    "グレネード配備",
    "グレネード配置",
    "グレネード配置完了",
    "セントリー配置",
    "セントリー設置",
    "タレット展開",
    "サイバーケージ設置",
    "スパイクを設置",
    "ドローン展開",
    "残り1名",
    "敵残り1名",
    "最後の一人だ",
    "最後の1人だ",
    "お遊びはここまでだ",
    "対戦を確認",
    "マッチポイント",
    "オーバータイムだ",
    "ディフェンダーの勝利",
    "アタッカーの勝利",
    "アルティメットいけるぞ",
    "スティールビーコンだ",
    "Aでキャリアダウン",
    "Bでキャリアダウン",
    "Cでキャリアダウン",
    "中央でキャリアダウン",
    "スポーンでキャリアダウン",
)

VALORANT_CALLOUTS: frozenset[str] = frozenset(
    _normalize(text) for text in _RAW_VALORANT_CALLOUTS
)

GAME_CALLOUTS: dict[str, frozenset[str]] = {
    "valorant": VALORANT_CALLOUTS,
}

_GAME_ALIASES: dict[str, str] = {
    "valorant": "valorant",
    "valo": "valorant",
    "瓦": "valorant",
    "瓦罗兰特": "valorant",
}


def supported_games() -> list[str]:
    return sorted(_GAME_ALIASES)


def parse_banned_words(banned_str: str) -> list[str]:
    if not banned_str:
        return []
    return [w.strip() for w in banned_str.replace("，", ",").split(",") if w.strip()]


def parse_filter_games(games_str: str) -> list[str]:
    return parse_banned_words(games_str)


def resolve_games(names: list[str]) -> frozenset[str]:
    merged: set[str] = set()
    for name in names:
        raw = str(name or "").strip()
        if not raw:
            continue
        key = _GAME_ALIASES.get(raw.lower())
        if key is None:
            raise ValueError(
                f"不支持的游戏过滤: {raw!r}。当前支持: {', '.join(supported_games())}"
            )
        merged |= GAME_CALLOUTS[key]
    return frozenset(merged)


def is_game_callout(text: str, callouts: frozenset[str]) -> bool:
    return bool(callouts) and _normalize(text) in callouts


def clean_unicode(text: str) -> str:
    result = []
    for char in text:
        if 0xD800 <= ord(char) <= 0xDFFF:
            continue
        try:
            char.encode("utf-8")
        except UnicodeEncodeError:
            continue
        result.append(char)
    return "".join(result)


def filter_text(
    text: str, blacklist: list[str], game_callouts: frozenset[str] | None = None
) -> str | None:
    text = clean_unicode(text or "").strip()
    if not text:
        return None
    low = text.lower()
    if any(bad.lower() in low for bad in blacklist):
        return None
    if is_game_callout(text, game_callouts or frozenset()):
        return None
    if len(text) < 2 and not text.isalnum():
        return None
    if text[0] in "(（【[":
        return None
    return text
