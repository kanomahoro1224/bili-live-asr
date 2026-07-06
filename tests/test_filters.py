"""filters 模块单测：垃圾词切分、文本过滤、Unicode 清洗。"""

from livetrans.filters import clean_unicode, filter_text, parse_banned_words


def test_parse_banned_words_handles_both_commas():
    assert parse_banned_words("a, b，c") == ["a", "b", "c"]
    assert parse_banned_words("") == []


def test_filter_text_drops_banned():
    bl = parse_banned_words("Music, 字幕")
    assert filter_text("背景 Music 播放", bl) is None
    assert filter_text("これはテスト", bl) == "これはテスト"


def test_filter_text_drops_short_nonalnum_and_brackets():
    assert filter_text("。", []) is None
    assert filter_text("（拍手）", []) is None
    assert filter_text("[BGM]", []) is None
    assert filter_text("", []) is None


def test_filter_text_keeps_normal():
    assert filter_text("  おはよう  ", []) == "おはよう"


def test_clean_unicode_strips_surrogates():
    assert clean_unicode("ab\ud800c") == "abc"
