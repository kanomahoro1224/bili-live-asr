# bili-live-asr

Bilibili 直播同传控制台。本版本使用本地 ASR、Silero VAD、OpenAI 兼容翻译接口，并提供 Flask + Socket.IO Web 控制台。

## 功能

- 拉取 Bilibili 直播流并用本地 ASR 识别语音。
- 使用 LLM 将 ASR 输出翻译为中文。
- Web 页面实时显示 ASR 原文、TL 译文、ASR 耗时和 TL 耗时。
- 支持 Bilibili 扫码登录，登录 cookie 保存到 `bilicookie.json`。
- 支持手动发送弹幕、屏蔽词、VAD 参数和翻译参数配置。

## 运行

运行前需要先安装外部工具 `ffmpeg`，并确认它在 PATH 中：

```powershell
ffmpeg -version
ffprobe -version
```

`ffmpeg` 不放进 uv 环境，也不提交到仓库；建议用系统包管理器安装。

```powershell
uv sync
uv run python app.py
```

然后打开 `http://127.0.0.1:5000`。

## 本地文件

以下文件不会提交到仓库，需要按本机环境准备：

- `config.json`
- `bilicookie.json`
- `models/`
- `.venv/`
- `.uv-cache/`
- `output/`

模型建议放在 `models/<模型名>/`，例如 `models/kotoba-whisper-v2.2/`。
