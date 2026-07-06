"""文本过滤与清洗（纯逻辑，可独立单测）。

负责：垃圾词（banned_words）过滤、ASR 文本基础清洗、无效 Unicode 代理对剔除。
本地 ASR 单段产出一条文本，filter_text 判定其是否应保留。
"""

from __future__ import annotations

__all__ = [
    "parse_banned_words",
    "clean_unicode",
    "filter_text",
]


def parse_banned_words(banned_str: str) -> list[str]:
    """把配置里的垃圾词字符串切成列表（兼容中英文逗号）。"""
    if not banned_str:
        return []
    return [w.strip() for w in banned_str.replace("，", ",").split(",") if w.strip()]


def clean_unicode(text: str) -> str:
    """剔除无效的 Unicode 代理对字符（U+D800..U+DFFF）及不可编码字符。"""
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


def filter_text(text: str, blacklist: list[str]) -> str | None:
    """对一条 ASR 文本应用过滤规则；保留则返回清洗后的文本，否则返回 None。

    规则（沿用在线版 get_qwen_asr_results_filtered 的判定）：
    - 命中垃圾词（不区分大小写）→ 丢弃
    - 长度 < 2 且非字母数字 → 丢弃
    - 以 ( （ 【 [ 开头（多为音效/旁白标记）→ 丢弃
    """
    text = (text or "").strip()
    if not text:
        return None
    low = text.lower()
    if any(bad.lower() in low for bad in blacklist):
        return None
    if len(text) < 2 and not text.isalnum():
        return None
    if text[0] in "(（【[":
        return None
    return text
