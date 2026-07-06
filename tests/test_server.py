import os

from livetrans import server


def test_find_available_port_returns_int():
    port = server._find_available_port(0)

    assert isinstance(port, int)
    assert port >= 0


def test_setup_cache_env_points_to_project_models(monkeypatch, tmp_path):
    calls = []
    from livetrans import model_manager

    monkeypatch.setattr(model_manager, "apply_cache_env", lambda: calls.append(model_manager.MODELS_DIR))

    server._setup_cache_env(str(tmp_path))

    assert calls == [tmp_path / "models"]
