"""Flask routes and Socket.IO setup for the web console."""

from __future__ import annotations

import os
import time

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_socketio import SocketIO

from . import bili, security
from .config import DEFAULT_CONFIG, save_config
from .logging_util import log, log_buffer

__all__ = ["create_web"]

_rate_limiter = security.RateLimiter()
_login_guard = security.LoginGuard()


def create_web(state):
    """Build and return the Flask app and Socket.IO instance."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY", "default_secret_key_change_me_immediately"
    )
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    state.socketio = socketio
    config = state.config
    current_dir = state.current_dir
    cookie_file = os.path.join(current_dir, "bilicookie.json")

    @app.before_request
    def _filter_malicious():
        return security.filter_malicious_requests(
            config.get("log_security_events", False), log
        )

    @app.before_request
    def _require_login():
        if ".." in request.path or request.path.startswith("//"):
            return jsonify({"ok": False, "error": "非法路径"}), 403
        if request.method == "POST" and not _rate_limiter.allow(request.remote_addr):
            return jsonify({"ok": False, "error": "请求过于频繁，请稍后再试"}), 429
        allowed = ["login", "static", "danmu_css"]
        if request.endpoint not in allowed and not session.get("logged_in"):
            return redirect(url_for("login"))

    @app.route("/danmu.css")
    def danmu_css():
        return send_from_directory(current_dir, "danmu.css")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            ip = request.remote_addr
            remaining = _login_guard.locked_remaining(ip)
            if remaining > 0:
                return render_template(
                    "login.html",
                    error=f"登录失败次数过多，请等待 {remaining} 秒后重试",
                )
            password = request.form.get("password", "")
            if password == config.get("web_password", "admin"):
                _login_guard.record_success(ip)
                session["logged_in"] = True
                session.permanent = True
                return redirect("/")
            left = _login_guard.record_failure(ip)
            if left < 0:
                return render_template(
                    "login.html",
                    error=(
                        "登录失败次数过多，账号已锁定 "
                        f"{security.LOGIN_LOCKOUT_TIME // 60} 分钟"
                    ),
                )
            return render_template("login.html", error=f"密码错误，剩余尝试次数：{left}")
        return render_template("login.html")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/latest")
    def latest():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        with state.history_lock:
            return jsonify(list(state.history_buffer))

    @app.route("/status")
    def status():
        return jsonify({"running": state.is_running})

    @app.route("/theme", methods=["GET", "POST"])
    def theme():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        if request.method == "GET":
            with state.config_lock:
                return jsonify(
                    {"ok": True, "theme_color": config.get("theme_color", "#4f46e5")}
                )
        color = str((request.json or {}).get("theme_color", "")).strip()
        is_hex = (
            len(color) == 7
            and color.startswith("#")
            and all(ch in "0123456789abcdefABCDEF" for ch in color[1:])
        )
        if not is_hex:
            return jsonify({"ok": False, "error": "无效主题颜色"}), 400
        with state.config_lock:
            config["theme_color"] = color
            save_config(state.config_path, config)
        return jsonify({"ok": True, "theme_color": color})

    @app.route("/subtitle/mode", methods=["GET", "POST"])
    def subtitle_mode():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        if request.method == "GET":
            with state.config_lock:
                return jsonify(
                    {
                        "ok": True,
                        "mode": config.get("subtitle_send_mode", "manual"),
                        "min_interval": config.get("subtitle_min_interval", 2.0),
                    }
                )
        mode = str((request.json or {}).get("mode", "")).strip()
        if mode not in {"manual", "auto"}:
            return jsonify({"ok": False, "error": "无效字幕模式"}), 400
        with state.config_lock:
            config["subtitle_send_mode"] = mode
            config["subtitle_min_interval"] = 2.0
            save_config(state.config_path, config)
        log("Subtitle", f"字幕发送模式: {mode}")
        return jsonify({"ok": True, "mode": mode, "min_interval": 2.0})

    @app.route("/asr/devices")
    def asr_devices():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        devices = [{"value": "cuda", "label": "CUDA 默认 GPU"}, {"value": "cpu", "label": "CPU"}]
        try:
            import torch

            if torch.cuda.is_available():
                devices = [{"value": "cuda", "label": "CUDA 默认 GPU"}]
                for index in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(index)
                    devices.append({"value": f"cuda:{index}", "label": f"GPU {index}: {name}"})
                devices.append({"value": "cpu", "label": "CPU"})
        except Exception:
            pass
        return jsonify({"ok": True, "devices": devices})

    @app.route("/logs")
    def get_logs():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        return jsonify({"logs": list(log_buffer)})

    @app.route("/bili/profile")
    def bili_profile():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        with state.config_lock:
            room_id = config.get("bili_room_id", "")
        return jsonify(
            {
                "ok": True,
                "account": bili.get_account_profile(cookie_file),
                "room": bili.get_room_profile(room_id),
                "room_id": room_id,
            }
        )

    @app.route("/bili/room", methods=["POST"])
    def bili_room():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        room_id = str((request.json or {}).get("room_id", "")).strip()
        if not room_id.isdigit():
            return jsonify({"ok": False, "error": "房间号只能是数字"}), 400
        with state.config_lock:
            config["bili_room_id"] = room_id
            config["bili_room_url"] = f"https://live.bilibili.com/{room_id}"
            save_config(state.config_path, config)
        state.stream_reload_event.set()
        return jsonify(
            {
                "ok": True,
                "room_id": room_id,
                "room": bili.get_room_profile(room_id),
            }
        )

    @app.route("/bili/avatar")
    def bili_avatar():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        try:
            body, content_type = bili.fetch_avatar_image(request.args.get("url", ""))
        except Exception:
            return jsonify({"ok": False, "error": "头像加载失败"}), 404
        return Response(
            body,
            content_type=content_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.route("/bili/login/qrcode", methods=["POST"])
    def bili_login_qrcode():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        try:
            data = bili.create_login_qrcode()
            return jsonify({"ok": True, **data})
        except Exception as e:
            log("Bili", f"二维码创建失败: {e}")
            return jsonify({"ok": False, "error": str(e)}), 502

    @app.route("/bili/login/poll", methods=["POST"])
    def bili_login_poll():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        key = (request.json or {}).get("qrcode_key", "")
        try:
            return jsonify(bili.poll_login_qrcode(key, cookie_file))
        except Exception as e:
            log("Bili", f"二维码登录轮询失败: {e}")
            return jsonify({"ok": False, "status": "error", "message": str(e)}), 502

    @app.route("/bili/logout", methods=["POST"])
    def bili_logout():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        try:
            bili.clear_cookie_file(cookie_file)
            legacy_cookie_file = os.path.join(current_dir, "bilicookies.json")
            if legacy_cookie_file != cookie_file and os.path.exists(legacy_cookie_file):
                bili.clear_cookie_file(legacy_cookie_file)
        except Exception as e:
            log("Bili", f"退出登录失败: {e}")
            return jsonify({"ok": False, "error": "退出登录失败"}), 500
        log("Bili", "已退出 Bilibili 账号")
        return jsonify(
            {
                "ok": True,
                "account": {"logged_in": False, "name": "点击登录", "avatar": ""},
            }
        )

    @app.route("/export/<fmt>")
    def export_file(fmt):
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        date_str = request.args.get("date", time.strftime("%Y-%m-%d"))
        if ".." in date_str or "/" in date_str or "\\" in date_str:
            return jsonify({"ok": False, "error": "非法日期"}), 400
        if fmt not in ("csv", "json"):
            return jsonify({"ok": False, "error": "不支持的格式"}), 400
        save_dir = os.path.join(current_dir, "output")
        filepath = os.path.join(save_dir, f"{date_str}.{fmt}")
        if not os.path.exists(filepath):
            return jsonify({"ok": False, "error": f"没有 {date_str} 的记录"}), 404
        return send_from_directory(save_dir, os.path.basename(filepath), as_attachment=True)

    @app.route("/toggle_run", methods=["POST"])
    def toggle_run():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        action = (request.json or {}).get("action")
        if action == "start":
            state.is_running = True
        elif action == "stop":
            state.is_running = False
        else:
            return jsonify({"ok": False, "error": "无效操作"}), 400
        log("Ctrl", f"用户操作: {action}")
        socketio.emit("status_update", {"running": state.is_running})
        return jsonify({"running": state.is_running})

    @app.route("/settings", methods=["GET", "POST"])
    def settings_api():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        if request.method == "GET":
            with state.config_lock:
                return jsonify(dict(config))
        new_data = request.json or {}
        allowed = set(DEFAULT_CONFIG.keys())
        reload_asr = False
        with state.config_lock:
            for k, v in new_data.items():
                if k not in allowed:
                    continue
                default = DEFAULT_CONFIG[k]
                if isinstance(default, bool):
                    v = bool(v)
                elif isinstance(default, (int, float)):
                    try:
                        v = type(default)(v)
                    except (ValueError, TypeError):
                        continue
                elif isinstance(default, str):
                    if not (isinstance(v, str) and len(v) < 10000):
                        continue
                if k == "max_record_time":
                    config["max_speech_duration"] = v
                if _is_asr_reload_key(k) and v != config.get(k):
                    reload_asr = True
                config[k] = v
            save_config(state.config_path, config)
        if reload_asr:
            state.reload_event.set()
            log("Config", "ASR/VAD 配置已更新，将在下次循环重载")
        return jsonify({"ok": True})

    @app.route("/send", methods=["POST"])
    def send():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        text = (request.json or {}).get("text", "")
        if not text or not text.strip():
            return jsonify({"ok": False, "error": "内容为空"})
        if len(text) > 200:
            return jsonify({"ok": False, "error": "内容过长，最多 200 字符"})
        with state.config_lock:
            send_config = dict(config)
        return jsonify(security.send_danmu(text, send_config, cookie_file))

    return app, socketio


def _is_asr_reload_key(key: str) -> bool:
    return key in {
        "asr_engine",
        "asr_model_id",
        "asr_device",
        "asr_language",
        "asr_num_beams",
        "asr_chunk_length_s",
        "asr_batch_size",
        "asr_stride_left_s",
        "asr_stride_right_s",
        "dashscope_api_key",
        "remote_asr_model",
        "remote_asr_timeout",
        "remote_realtime_asr_model",
        "remote_realtime_asr_url",
        "remote_realtime_asr_timeout",
        "vad_threshold",
        "vad_device",
        "min_speech_duration",
        "max_record_time",
        "max_speech_duration",
        "silence_mode",
        "silence_duration",
    }
