import json
from types import SimpleNamespace

import pytest
import requests

from livetrans import bili


def test_cookie_helpers_support_list_and_dict(tmp_path):
    cookie_file = tmp_path / "bilicookie.json"
    cookie_file.write_text(
        json.dumps([{"name": "SESSDATA", "value": "abc"}, {"name": "bili_jct", "value": "csrf"}]),
        encoding="utf-8",
    )

    assert bili.load_cookie_items(str(cookie_file))[0]["name"] == "SESSDATA"
    assert bili.build_cookie_header(str(cookie_file)) == "SESSDATA=abc; bili_jct=csrf"

    cookie_file.write_text(json.dumps({"bili_cookie": "a=1; b=2"}), encoding="utf-8")
    assert bili.load_cookie_items(str(cookie_file)) == [
        {"name": "a", "value": "1"},
        {"name": "b", "value": "2"},
    ]


def test_save_cookie_items_writes_cookie_jar(tmp_path, monkeypatch):
    monkeypatch.setattr(bili.time, "time", lambda: 123)
    jar = requests.cookies.RequestsCookieJar()
    jar.set("SESSDATA", "abc", domain=".bilibili.com", path="/")

    cookie_file = tmp_path / "bilicookie.json"
    bili.save_cookie_items(str(cookie_file), jar)

    data = json.loads(cookie_file.read_text(encoding="utf-8"))
    assert data[0]["name"] == "SESSDATA"
    assert data[0]["saved_at"] == 123


def test_clear_cookie_file_writes_empty_json(tmp_path):
    cookie_file = tmp_path / "bilicookie.json"
    cookie_file.write_text(json.dumps([{"name": "SESSDATA", "value": "abc"}]), encoding="utf-8")

    bili.clear_cookie_file(str(cookie_file))

    assert json.loads(cookie_file.read_text(encoding="utf-8")) == []
    assert bili.build_cookie_header(str(cookie_file)) == ""


def test_fetch_avatar_image_validates_host_and_content_type(monkeypatch):
    class Resp:
        content = b"img"
        headers = {"Content-Type": "image/png; charset=utf-8"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(bili.requests, "get", lambda *args, **kwargs: Resp())

    body, content_type = bili.fetch_avatar_image("//i0.hdslb.com/a.png")

    assert body == b"img"
    assert content_type == "image/png"
    with pytest.raises(ValueError):
        bili.fetch_avatar_image("https://example.com/a.png")


def test_profile_helpers_return_defaults_or_api_data(monkeypatch, tmp_path):
    cookie_file = tmp_path / "bilicookie.json"
    assert bili.get_account_profile(str(cookie_file))["name"] == "点击登录"

    cookie_file.write_text(json.dumps({"bili_cookie": "SESSDATA=abc"}), encoding="utf-8")

    def fake_get_json(url, cookie_file=None, timeout=10):
        if "nav" in url:
            return {"code": 0, "data": {"isLogin": True, "uname": "user", "face": "face", "mid": 1}}
        return {"code": 0, "data": {"info": {"uname": "room", "face": "avatar", "title": "title"}}}

    monkeypatch.setattr(bili, "_get_json", fake_get_json)

    assert bili.get_account_profile(str(cookie_file))["name"] == "user"
    assert bili.get_room_profile("123")["name"] == "room"


def test_create_and_poll_login_qrcode(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bili,
        "_get_json",
        lambda url, cookie_file=None, timeout=10: {
            "code": 0,
            "data": {"url": "https://passport.bilibili.com/q", "qrcode_key": "key"},
        },
    )
    data = bili.create_login_qrcode()
    assert data["qrcode_key"] == "key"
    assert data["image"].startswith("data:image/png;base64,")

    class Resp:
        cookies = requests.cookies.RequestsCookieJar()

        def __init__(self):
            self.cookies.set("SESSDATA", "abc", domain=".bilibili.com", path="/")

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"code": 86090}}

    monkeypatch.setattr(bili.requests, "get", lambda *args, **kwargs: Resp())

    result = bili.poll_login_qrcode("key", str(tmp_path / "cookie.json"))
    assert result["status"] == "scanned"

    assert bili.poll_login_qrcode("", str(tmp_path / "cookie.json"))["status"] == "missing_key"
