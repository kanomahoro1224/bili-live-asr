import json

from livetrans.config import DEFAULT_CONFIG, save_config
from livetrans.state import AppState
from livetrans.web import _is_asr_reload_key, create_web


def make_client(tmp_path, monkeypatch):
    config = DEFAULT_CONFIG.copy()
    config["web_password"] = "pw"
    config_path = tmp_path / "config.json"
    save_config(str(config_path), config)
    state = AppState(config, str(config_path), str(tmp_path))
    app, _ = create_web(state)
    app.config["TESTING"] = True
    monkeypatch.setattr("livetrans.web.bili.get_room_profile", lambda room_id: {"name": "room"})
    monkeypatch.setattr(
        "livetrans.web.bili.get_account_profile",
        lambda cookie_file: {"logged_in": False, "name": "点击登录", "avatar": ""},
    )
    return app.test_client(), state, config_path


def login(client):
    with client.session_transaction() as sess:
        sess["logged_in"] = True


def test_login_and_status_routes(tmp_path, monkeypatch):
    client, state, _ = make_client(tmp_path, monkeypatch)

    resp = client.post("/login", data={"password": "pw"})
    assert resp.status_code == 302
    assert client.get("/").status_code == 200
    assert client.get("/status").json == {"running": False}


def test_theme_subtitle_mode_and_settings_save(tmp_path, monkeypatch):
    client, state, config_path = make_client(tmp_path, monkeypatch)
    login(client)

    assert client.post("/theme", json={"theme_color": "#123abc"}).json == {
        "ok": True,
        "theme_color": "#123abc",
    }
    assert state.config["theme_color"] == "#123abc"
    assert client.post("/theme", json={"theme_color": "bad"}).status_code == 400

    assert client.post("/subtitle/mode", json={"mode": "auto"}).json["mode"] == "auto"
    assert state.config["subtitle_min_interval"] == 2.0
    assert client.post("/subtitle/mode", json={"mode": "bad"}).status_code == 400

    resp = client.post(
        "/settings",
        json={
            "asr_language": "ja",
            "vad_threshold": "0.3",
            "tl_timeout": "12.5",
            "translation_model_type": "qwen_mt",
            "qwen_mt_base_url": "https://qwen.example/v1",
            "qwen_mt_api_key": "qwen-key",
            "qwen_mt_model": "qwen-mt-flash",
            "qwen_mt_source_lang": "Japanese",
            "qwen_mt_target_lang": "Chinese",
            "qwen_mt_terms_enabled": True,
            "qwen_mt_terms": [{"source": "鹿乃", "target": "Kano"}],
            "qwen_mt_tm_list_enabled": True,
            "qwen_mt_tm_list": [{"source": "おはよう", "target": "早上好"}],
            "qwen_mt_domains_enabled": True,
            "qwen_mt_domains": "Translate into a casual livestream subtitle style.",
            "unknown": "ignored",
        },
    )
    assert resp.json == {"ok": True}
    assert state.config["vad_threshold"] == 0.3
    assert state.config["tl_timeout"] == 12.5
    assert state.config["translation_model_type"] == "qwen_mt"
    assert "unknown" not in state.config
    assert state.reload_event.is_set()
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    assert loaded["vad"]["vad_threshold"] == 0.3
    assert loaded["translation"]["tl_timeout"] == 12.5
    assert loaded["translation"]["translation_model_type"] == "qwen_mt"
    assert "qwen_mt_base_url" not in loaded["translation"]
    assert loaded["qwen_mt"]["qwen_mt_base_url"] == "https://qwen.example/v1"
    assert loaded["qwen_mt"]["qwen_mt_api_key"] == "qwen-key"
    assert loaded["qwen_mt"]["qwen_mt_model"] == "qwen-mt-flash"
    assert loaded["qwen_mt"]["qwen_mt_source_lang"] == "Japanese"
    assert loaded["qwen_mt"]["qwen_mt_target_lang"] == "Chinese"
    assert loaded["qwen_mt"]["qwen_mt_terms_enabled"] is True
    assert loaded["qwen_mt"]["qwen_mt_terms"] == [{"source": "鹿乃", "target": "Kano"}]
    assert loaded["qwen_mt"]["qwen_mt_tm_list_enabled"] is True
    assert loaded["qwen_mt"]["qwen_mt_tm_list"] == [
        {"source": "おはよう", "target": "早上好"}
    ]
    assert loaded["qwen_mt"]["qwen_mt_domains_enabled"] is True
    assert loaded["qwen_mt"]["qwen_mt_domains"] == "Translate into a casual livestream subtitle style."


def test_bili_room_toggle_send_and_export(tmp_path, monkeypatch):
    client, state, _ = make_client(tmp_path, monkeypatch)
    login(client)
    monkeypatch.setattr("livetrans.web.security.send_danmu", lambda text, config, cookie_file: {"ok": True})

    assert client.post("/bili/room", json={"room_id": "abc"}).status_code == 400
    assert client.post("/bili/room", json={"room_id": "123"}).json["room_id"] == "123"
    assert state.stream_reload_event.is_set()

    assert client.post("/toggle_run", json={"action": "start"}).json == {"running": True}
    assert state.is_running
    assert client.post("/toggle_run", json={"action": "bad"}).status_code == 400

    assert client.post("/send", json={"text": ""}).json["ok"] is False
    assert client.post("/send", json={"text": "你好"}).json == {"ok": True}

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "2026-01-02.json").write_text("[]", encoding="utf-8")
    assert client.get("/export/json?date=2026-01-02").status_code == 200
    assert client.get("/export/json?date=../x").status_code == 400
    assert client.get("/export/txt?date=2026-01-02").status_code == 400


def test_profile_avatar_and_qrcode_routes(tmp_path, monkeypatch):
    client, state, _ = make_client(tmp_path, monkeypatch)
    login(client)
    monkeypatch.setattr("livetrans.web.bili.fetch_avatar_image", lambda url: (b"img", "image/png"))
    monkeypatch.setattr("livetrans.web.bili.create_login_qrcode", lambda: {"qrcode_key": "k", "image": "data"})
    monkeypatch.setattr(
        "livetrans.web.bili.poll_login_qrcode",
        lambda key, cookie_file: {"ok": False, "status": "waiting"},
    )

    assert client.get("/bili/profile").json["ok"] is True
    assert client.get("/bili/avatar?url=https://i0.hdslb.com/a.png").data == b"img"
    assert client.post("/bili/login/qrcode").json["qrcode_key"] == "k"
    assert client.post("/bili/login/poll", json={"qrcode_key": "k"}).json["status"] == "waiting"


def test_bili_logout_clears_cookie_files(tmp_path, monkeypatch):
    client, state, _ = make_client(tmp_path, monkeypatch)
    login(client)
    cookie_file = tmp_path / "bilicookie.json"
    legacy_cookie_file = tmp_path / "bilicookies.json"
    cookie_file.write_text(json.dumps([{"name": "SESSDATA", "value": "abc"}]), encoding="utf-8")
    legacy_cookie_file.write_text(json.dumps([{"name": "SESSDATA", "value": "old"}]), encoding="utf-8")

    data = client.post("/bili/logout").json

    assert data["ok"] is True
    assert data["account"]["logged_in"] is False
    assert json.loads(cookie_file.read_text(encoding="utf-8")) == []
    assert json.loads(legacy_cookie_file.read_text(encoding="utf-8")) == []


def test_is_asr_reload_key():
    assert _is_asr_reload_key("asr_language")
    assert _is_asr_reload_key("vad_threshold")
    assert not _is_asr_reload_key("theme_color")
