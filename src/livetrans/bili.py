"""Bilibili account helpers: QR login, cookie storage, and profile lookup."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import qrcode
import requests

__all__ = [
    "build_cookie_header",
    "load_cookie_items",
    "save_cookie_items",
    "get_account_profile",
    "get_room_profile",
    "fetch_avatar_image",
    "create_login_qrcode",
    "poll_login_qrcode",
]

USER_AGENT = "Mozilla/5.0"
DEFAULT_AVATAR = ""
ALLOWED_IMAGE_HOST_SUFFIXES = (".hdslb.com", ".bilibili.com")


def load_cookie_items(cookie_file: str) -> list[dict[str, Any]]:
    if not os.path.exists(cookie_file):
        return []
    try:
        with open(cookie_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and data.get("bili_cookie"):
        items = []
        for part in str(data["bili_cookie"]).split(";"):
            if "=" not in part:
                continue
            name, value = part.strip().split("=", 1)
            items.append({"name": name, "value": value})
        return items
    return []


def build_cookie_header(cookie_file: str) -> str:
    items = load_cookie_items(cookie_file)
    return "; ".join(
        f"{item['name']}={item['value']}"
        for item in items
        if item.get("name") and item.get("value") is not None
    )


def save_cookie_items(cookie_file: str, cookies: requests.cookies.RequestsCookieJar) -> None:
    items = []
    for cookie in cookies:
        items.append(
            {
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path,
                "expires": cookie.expires,
                "saved_at": int(time.time()),
            }
        )
    if not items:
        raise RuntimeError("登录成功但未收到 Cookie")
    tmp = f"{cookie_file}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cookie_file)


def _get_json(url: str, cookie_file: str | None = None, timeout: int = 10) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"}
    if cookie_file:
        cookie = build_cookie_header(cookie_file)
        if cookie:
            headers["Cookie"] = cookie
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _normalize_image_url(url: str) -> str:
    url = (url or "").strip()
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https":
        raise ValueError("unsupported image url")
    if not any(host == suffix[1:] or host.endswith(suffix) for suffix in ALLOWED_IMAGE_HOST_SUFFIXES):
        raise ValueError("unsupported image host")
    return url


def fetch_avatar_image(url: str) -> tuple[bytes, str]:
    url = _normalize_image_url(url)
    resp = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": "https://www.bilibili.com/",
        },
        timeout=10,
    )
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";", 1)[0]
    if not content_type.startswith("image/"):
        raise ValueError("url did not return an image")
    return resp.content, content_type


def get_account_profile(cookie_file: str) -> dict[str, Any]:
    cookie = build_cookie_header(cookie_file)
    if not cookie:
        return {"logged_in": False, "name": "点击登录", "avatar": DEFAULT_AVATAR}
    try:
        data = _get_json("https://api.bilibili.com/x/web-interface/nav", cookie_file)
        info = data.get("data") or {}
        if data.get("code") == 0 and info.get("isLogin"):
            return {
                "logged_in": True,
                "name": info.get("uname") or "已登录",
                "avatar": info.get("face") or DEFAULT_AVATAR,
                "mid": info.get("mid"),
            }
    except Exception:
        pass
    return {"logged_in": False, "name": "点击登录", "avatar": DEFAULT_AVATAR}


def get_room_profile(room_id: str) -> dict[str, Any]:
    if not room_id:
        return {"name": "", "avatar": DEFAULT_AVATAR, "title": ""}
    try:
        data = _get_json(
            "https://api.live.bilibili.com/live_user/v1/UserInfo/get_anchor_in_room"
            f"?roomid={room_id}"
        )
        info = ((data.get("data") or {}).get("info") or {})
        if data.get("code") == 0 and info:
            return {
                "name": info.get("uname") or "",
                "avatar": info.get("face") or DEFAULT_AVATAR,
                "title": info.get("title") or "",
            }
    except Exception:
        pass
    try:
        data = _get_json(f"https://api.live.bilibili.com/room/v1/Room/get_info?room_id={room_id}")
        info = data.get("data") or {}
        if data.get("code") == 0:
            return {
                "name": info.get("uname") or "",
                "avatar": info.get("user_cover") or DEFAULT_AVATAR,
                "title": info.get("title") or "",
            }
    except Exception:
        pass
    return {"name": "", "avatar": DEFAULT_AVATAR, "title": ""}


def create_login_qrcode() -> dict[str, str]:
    data = _get_json("https://passport.bilibili.com/x/passport-login/web/qrcode/generate")
    payload = data.get("data") or {}
    url = payload.get("url")
    key = payload.get("qrcode_key")
    if data.get("code") != 0 or not url or not key:
        raise RuntimeError(data.get("message") or "无法创建登录二维码")
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"qrcode_key": key, "image": f"data:image/png;base64,{qr_data}"}


def poll_login_qrcode(qrcode_key: str, cookie_file: str) -> dict[str, Any]:
    if not qrcode_key:
        return {"ok": False, "status": "missing_key", "message": "缺少二维码状态"}
    resp = requests.get(
        "https://passport.bilibili.com/x/passport-login/web/qrcode/poll",
        params={"qrcode_key": qrcode_key},
        headers={"User-Agent": USER_AGENT, "Referer": "https://www.bilibili.com/"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    payload = data.get("data") or {}
    code = payload.get("code")
    if code == 0:
        save_cookie_items(cookie_file, resp.cookies)
        return {
            "ok": True,
            "status": "confirmed",
            "account": get_account_profile(cookie_file),
        }
    status_map = {
        86101: ("waiting", "等待扫码"),
        86090: ("scanned", "已扫码，请在手机上确认"),
        86038: ("expired", "二维码已过期"),
    }
    status, message = status_map.get(code, ("unknown", payload.get("message") or data.get("message") or "登录状态未知"))
    return {"ok": False, "status": status, "message": message}
