import os

from livetrans import model_manager as mm


def test_funasr_selection_and_ids():
    assert mm.normalize_funasr_model_key("sensevoice") == "sensevoice-small"
    assert mm.normalize_asr_engine_selection("sensevoice") == ("funasr", "sensevoice-small")
    assert mm.migrate_funasr_settings({"asr_engine": "funasr-nano"}) == {
        "asr_engine": "funasr",
        "funasr_model": "funasr-nano-2512",
    }
    assert mm.asr_model_id("sensevoice", "hf") == "FunAudioLLM/SenseVoiceSmall"
    assert mm.funasr_model_id("sensevoice-small", "ms") == "iic/SenseVoiceSmall"
    assert mm.funasr_supports_padding("sensevoice-small")


def test_cache_env_and_format_size(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path / "models")

    mm.apply_cache_env()

    assert os.environ["MODELSCOPE_CACHE"].endswith(os.path.join("models", "modelscope"))
    assert os.environ["HF_HOME"].endswith(os.path.join("models", "huggingface"))
    assert os.environ["TORCH_HOME"].endswith(os.path.join("models", "torch"))
    assert mm.format_size(999) == "999 B"
    assert mm.format_size(2048) == "2.0 KB"


def test_faster_whisper_local_model_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "APP_DIR", tmp_path)
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path / "models")
    model_dir = tmp_path / "models" / "custom"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"1")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    assert mm.is_faster_whisper_model_dir(model_dir)
    assert mm.resolve_custom_whisper_model(str(model_dir)) == str(model_dir.resolve())
    assert mm.list_local_faster_whisper_models() == [
        {"name": "custom", "path": str(model_dir.resolve())}
    ]
    assert mm.local_faster_whisper_display_name(str(model_dir)) == "custom"


def test_hf_snapshot_name_and_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path / "models")
    snap = (
        tmp_path
        / "models"
        / "huggingface"
        / "hub"
        / "models--Org--Repo"
        / "snapshots"
        / "abc"
    )
    snap.mkdir(parents=True)
    (snap / "weights.bin").write_bytes(b"12345")

    assert mm._hf_snapshot_name(snap) == "Org/Repo"
    assert mm._hf_repo_complete("Org", "Repo", min_bytes=5)


def test_neutralize_requirements_and_dir_size(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("gradio", encoding="utf-8")
    data = tmp_path / "data.bin"
    data.write_bytes(b"123")

    mm.neutralize_funasr_requirements(tmp_path)

    assert not req.exists()
    assert (tmp_path / "requirements.txt.bundled").exists()
    assert mm.dir_size(tmp_path) >= 3
