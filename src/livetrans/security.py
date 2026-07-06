"""Web 安全防护与 Bilibili 弹幕（依赖 flask request + requests）。

- filter_malicious_requests: 拦截非 HTTP 协议探测（TLS/SSH/Oracle/T3/RDP）与扫描器 UA。
- RateLimiter / LoginGuard: POST 限频、登录失败锁定（纯逻辑，可单测）。
- get_bili_creds / send_danmu: 读取 B 站 cookie/csrf 并发送弹幕。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from flask import request

__all__ = [
    "MAX_LOGIN_ATTEMPTS",
    "LOGIN_LOCKOUT_TIME",
    "MAX_REQUESTS_PER_MINUTE",
    "RateLimiter",
    "LoginGuard",
    "filter_malicious_requests",
    "get_bili_creds",
    "send_danmu",
]

MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_TIME = 300  # 秒
MAX_REQUESTS_PER_MINUTE = 60


class RateLimiter:
    """按 IP 的滑动窗口 POST 限频。"""

    def __init__(self, max_per_minute: int = MAX_REQUESTS_PER_MINUTE):
        self.max_per_minute = max_per_minute
        self._times: dict[str, list[float]] = {}

    def allow(self, ip: str, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        times = [t for t in self._times.get(ip, []) if now - t < 60]
        if len(times) >= self.max_per_minute:
            self._times[ip] = times
            return False
        times.append(now)
        self._times[ip] = times
        return True


class LoginGuard:
    """登录失败计数与锁定。check() 返回剩余锁定秒数(>0 表示锁定中)。"""

    def __init__(self, max_attempts=MAX_LOGIN_ATTEMPTS, lockout=LOGIN_LOCKOUT_TIME):
        self.max_attempts = max_attempts
        self.lockout = lockout
        self._attempts: dict[str, list] = {}  # ip -> [count, lock_until|None]

    def locked_remaining(self, ip: str, now: float | None = None) -> int:
        now = now if now is not None else time.time()
        rec = self._attempts.get(ip)
        if rec and rec[1] and now < rec[1]:
            return int(rec[1] - now)
        if rec and rec[1] and now >= rec[1]:
            self._attempts[ip] = [0, None]
        return 0

    def record_success(self, ip: str) -> None:
        self._attempts.pop(ip, None)

    def record_failure(self, ip: str, now: float | None = None) -> int:
        """记一次失败，返回剩余可试次数；锁定时返回 -1。"""
        now = now if now is not None else time.time()
        rec = self._attempts.setdefault(ip, [0, None])
        rec[0] += 1
        if rec[0] >= self.max_attempts:
            rec[1] = now + self.lockout
            return -1
        return self.max_attempts - rec[0]


def filter_malicious_requests(log_events: bool, log_fn) -> tuple[str, int] | None:
    """拦截协议探测与扫描器；命中返回 (body, status)，否则 None。"""
    try:
        wreq = request.environ.get("werkzeug.request")
        if wreq is not None:
            raw = getattr(wreq, "data", b"")
            if raw:
                if raw[:2] in (b"\x16\x03", b"\x16\x02"):
                    if log_events:
                        log_fn("Security", f"阻止 TLS 探测: {request.remote_addr}")
                    return "This is an HTTP server, not HTTPS", 400
                if raw.startswith(b"SSH-"):
                    return "This is not an SSH server", 400
                if b"DESCRIPTION=" in raw and b"CONNECT_DATA=" in raw:
                    return "This is not a database server", 400
                if raw.startswith(b"t3 "):
                    return "This is not a WebLogic server", 400
                if b"Cookie: mstshash=" in raw:
                    return "This is not an RDP server", 400
        ua = request.headers.get("User-Agent", "").lower()
        if any(s in ua for s in ("masscan", "nmap", "nikto", "sqlmap", "scanner")):
            if log_events:
                log_fn("Security", f"阻止可疑 UA: {request.remote_addr} - {ua}")
            return "Forbidden", 403
    except Exception:
        pass
    return None


def get_bili_creds(config: dict[str, Any], cookie_file: str) -> tuple[str, str]:
    """读取 cookie/csrf：优先 bilicookie.json（数组或对象），回退 config。"""
    cookie = config.get("bili_cookie", "")
    csrf = config.get("bili_csrf", "")
    if os.path.exists(cookie_file):
        try:
            with open(cookie_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                parts = []
                for item in data:
                    parts.append(f"{item['name']}={item['value']}")
                    if item["name"] == "bili_jct":
                        csrf = item["value"]
                cookie = "; ".join(parts)
            elif isinstance(data, dict):
                cookie = data.get("bili_cookie", cookie)
                csrf = data.get("bili_csrf", csrf)
        except Exception:
            pass
    return cookie, csrf


def send_danmu(msg: str, config: dict[str, Any], cookie_file: str) -> dict[str, Any]:
    """向配置的直播间发送一条弹幕。"""
    cookie, csrf = get_bili_creds(config, cookie_file)
    room_id = config.get("bili_room_id")
    if not (cookie and csrf and room_id):
        return {"ok": False, "error": "Cookie/RoomID缺失"}
    payload = {
        "bubble": 0, "msg": msg, "color": 16777215, "mode": 1, "fontsize": 25,
        "rnd": int(time.time()), "roomid": int(room_id),
        "csrf": csrf, "csrf_token": csrf,
    }
    headers = {
        "Cookie": cookie,
        "Referer": config.get("bili_room_url"),
        "User-Agent": "Mozilla/5.0",
    }
    try:
        r = requests.post(
            "https://api.live.bilibili.com/msg/send",
            data=payload, headers=headers, timeout=10,
        )
        res = r.json()
        if res.get("code") == 0:
            return {"ok": True}
        return {"ok": False, "error": res.get("message")}
    except Exception as e:
        return {"ok": False, "error": str(e)}
