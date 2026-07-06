from livetrans import prompt_loader


def test_render_prompt_keeps_unknown_placeholders(tmp_path, monkeypatch):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "demo.txt").write_text("你好 {person} {missing}", encoding="utf-8")
    monkeypatch.setattr(prompt_loader, "PROMPT_DIR", prompt_dir)

    assert prompt_loader.render_prompt("demo.txt", person="鹿乃") == "你好 鹿乃 {missing}"
