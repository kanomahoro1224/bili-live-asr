import json
from http import HTTPStatus

from livetrans.security import (
    LoginGuard,
    RateLimiter,
    filter_malicious_requests,
    get_bili_creds,
    send_danmu,
)


def test_rate_limiter_sliding_window():
    limiter = RateLimiter(max_per_minute=2)

    assert limiter.allow("1.2.3.4", now=0)
    assert limiter.allow("1.2.3.4", now=1)
    assert not limiter.allow("1.2.3.4", now=2)
    assert limiter.allow("1.2.3.4", now=61)


def test_login_guard_locks_and_resets():
    guard = LoginGuard(max_attempts=2, lockout=10)

    assert guard.record_failure("ip", now=100) == 1
    assert guard.record_failure("ip", now=101) == -1
    assert guard.locked_remaining("ip", now=105) == 6
    assert guard.locked_remaining("ip", now=112) == 0
    guard.record_success("ip")
    assert guard.locked_remaining("ip", now=113) == 0


def test_filter_malicious_requests_blocks_scanner_user_agent():
    class Req:
        remote_addr = "127.0.0.1"
        headers = {"User-Agent": "sqlmap"}
        environ = {}

    logs = []
    from livetrans import security

    old_request = security.request
    security.request = Req()
    try:
        assert filter_malicious_requests(True, lambda tag, msg: logs.append((tag, msg))) == (
            "Forbidden",
            403,
        )
    finally:
        security.request = old_request
    assert logs and logs[0][0] == "Security"


def test_get_bili_creds_reads_cookie_file(tmp_path):
    cookie_file = tmp_path / "bilicookie.json"
    cookie_file.write_text(
        json.dumps(
            [
                {"name": "SESSDATA", "value": "abc"},
                {"name": "bili_jct", "value": "csrf"},
            ]
        ),
        encoding="utf-8",
    )

    cookie, csrf = get_bili_creds({}, str(cookie_file))

    assert "SESSDATA=abc" in cookie
    assert csrf == "csrf"


def test_send_danmu_posts_payload(monkeypatch, tmp_path):
    cookie_file = tmp_path / "bilicookie.json"
    cookie_file.write_text(
        json.dumps({"bili_cookie": "SESSDATA=abc", "bili_csrf": "csrf"}),
        encoding="utf-8",
    )
    calls = {}

    class Resp:
        status_code = HTTPStatus.OK

        def json(self):
            return {"code": 0}

    def fake_post(url, data, headers, timeout):
        calls.update(url=url, data=data, headers=headers, timeout=timeout)
        return Resp()

    monkeypatch.setattr("livetrans.security.requests.post", fake_post)

    result = send_danmu(
        "你好",
        {"bili_room_id": "123", "bili_room_url": "https://live.bilibili.com/123"},
        str(cookie_file),
    )

    assert result == {"ok": True}
    assert calls["data"]["msg"] == "你好"
    assert calls["data"]["roomid"] == 123
    assert "SESSDATA=abc" in calls["headers"]["Cookie"]
