"""Flask routes and Socket.IO setup for the web console."""

from __future__ import annotations

import os
import time

from flask import (
    Flask,
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

    @app.route("/logs")
    def get_logs():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        return jsonify({"logs": list(log_buffer)})

    @app.route("/bili/profile")
    def bili_profile():
        if not session.get("logged_in"):
            return jsonify({"ok": False, "error": "未授权"}), 403
        return jsonify(
            {
                "ok": True,
                "account": bili.get_account_profile(cookie_file),
                "room": bili.get_room_profile(config.get("bili_room_id", "")),
            }
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
            return jsonify(config)
        new_data = request.json or {}
        allowed = set(DEFAULT_CONFIG.keys())
        reload_asr = False
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
        return jsonify(security.send_danmu(text, config, cookie_file))

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
        "vad_threshold",
        "min_speech_duration",
        "max_record_time",
        "max_speech_duration",
        "silence_mode",
        "silence_duration",
    }
