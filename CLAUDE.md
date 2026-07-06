# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

tong 是一个 Bilibili 直播实时同传控制台：解析直播流音频，用 `ffmpeg` 解码为 16k 单声道 PCM，经过 Silero VAD 切出语音段，再用本地 Kotoba Whisper ASR 识别日语，最后调用 OpenAI 兼容 LLM 翻译成中文，并通过 Flask + Socket.IO Web 控制台实时展示。沟通、文档和新增注释默认用中文。

根目录里的 `LiveTranslate-main/` 和其他 `*-main` 目录是参考项目，不是当前主入口。当前主代码在 `src/livetrans/`。

## 常用命令

依赖与运行统一走 `uv`：

```bash
uv sync --no-dev --inexact                 # 同步运行依赖
uv run --no-sync python app.py             # 启动 Web 控制台
uv run --no-sync python -m pytest tests -q # 跑当前项目单测
uv run --no-sync python -m compileall -q src
```

Windows 下也可以直接运行：

```bat
run.bat
```

注意：

- `run.bat` 会把 uv 缓存放到项目根目录的 `.uv-cache/`，避免大包占满系统盘。
- `torch` / `torchaudio` 在 Windows 下从 `pytorch-cu126` 索引安装，见 `pyproject.toml` 的 `[tool.uv.sources]`。不要随手改回 PyPI 默认源。
- ASR 模型是 `kotoba-tech/kotoba-whisper-v2.2`，下载到 `models/kotoba-whisper-v2.2`，规则是一个模型一个文件夹。
- `ffmpeg` 是运行时外部工具，须在 PATH 中；uv 不管理 ffmpeg 二进制，也不要把 `ffmpeg.exe` 提交进仓库。
- 当前根目录不是 Git 仓库；不要对参考目录或外部仓库做无关改动。

## 配置

运行配置在 `config.json`，文件按模块分组：

- `web`：控制台密码。
- `stream`：Bilibili 直播间 URL 和 room id。
- `bilibili_danmu`：可选弹幕发送 cookie / csrf。
- `asr`：ASR 引擎、模型 ID、模型目录、设备、语言和 Kotoba pipeline 参数。
- `vad`：Silero VAD 阈值、最短/最长语音段、静音切分参数和屏蔽词。
- `translation`：游戏/场景提示、额外提示词、OpenAI 兼容 LLM 的 `llm_base_url` / `llm_api_key` / `llm_model`、上下文窗口。
- `security`：基础请求过滤日志开关。

配置加载器在 `src/livetrans/config.py` 中。运行时代码仍使用扁平 dict，写盘时整理成分组 JSON。新增配置项时要同时更新 `DEFAULT_CONFIG` 和 `CONFIG_SECTIONS`。

## 架构

模块按“纯逻辑在下、重依赖延迟导入、编排在上”的方向组织：

- `app.py`：薄入口，把 `src/` 加入导入路径后调用 `livetrans.server.run()`。
- `server.py`：设置模型缓存环境，加载配置，创建 `AppState`，装配 Web 和后台流水线，启动 Socket.IO 服务。
- `state.py`：共享运行时状态容器，包括配置、历史记录、Socket.IO 实例、翻译器和重载事件。
- `web.py`：Flask 路由、登录、设置页、日志、导出、弹幕发送和内联前端模板。
- `pipeline.py`：实时处理主循环。直播流 URL → `audio.stream_frames()` → `vad.VADProcessor` → ASR → 过滤 → LLM 翻译 → Socket.IO 推送和落盘。
- `stream.py`：用 `streamlink` 从 Bilibili 直播间解析真实流地址。
- `audio.py`：调用 PATH 中的 `ffmpeg`，把直播流解码为 16kHz、单声道、512 samples 一帧的 float32 PCM。
- `vad.py`：Silero VAD 处理器。它只判断“是否有人声”和切分语音段，不做说话人识别。
- `asr.py`：本地 ASR。默认 `KotobaWhisperEngine` 使用 Transformers ASR pipeline；旧 `SenseVoiceEngine` 保留为可选实现。
- `llm.py`：OpenAI 兼容聊天客户端，只负责 base URL 规范化、鉴权头和 `/chat/completions` 请求。
- `translator.py`：翻译提示词、上下文窗口和输出行数清洗；HTTP 调用统一走 `llm.LLMClient`。
- `filters.py`：屏蔽词解析、Unicode 清理、ASR 文本过滤，纯逻辑可单测。
- `storage.py`：把原文/译文追加保存到 `output/YYYY-MM-DD.csv` 和 `.json`。
- `logging_util.py`：轻量日志缓冲，供 Web `/logs` 读取。

## 关键约定

- 只把 `src/livetrans/` 和根入口当作当前项目。`LiveTranslate-main/`、`submaku-stream-main/` 等目录只作参考。
- 重依赖必须延迟导入：`torch`、`transformers`、`funasr`、`silero_vad` 等不要在纯配置/过滤/存储模块顶层引入。
- ASR 模型目录遵循 `models/<模型名>`，例如 `models/kotoba-whisper-v2.2`。不要回退到默认 Hugging Face cache 作为主路径。
- LLM 调用统一走 `llm.LLMClient`。不要在 `translator.py` 或 Web 层重复写 HTTP 请求和鉴权逻辑。
- VAD 参数不要和 ASR 模型迁移混在一起改。用户明确要求“只搬 ASR”时，不改 VAD 阈值和切分策略。
- `config.json` 可能含真实 API Key / Cookie。不要在总结、日志或文档里复述完整密钥。
- 当前测试只覆盖 `tests/`；全量 pytest 可能误收集参考目录测试，优先运行 `uv run --no-sync python -m pytest tests -q`。

## 数据流

实时同传：

```text
Bilibili room URL
  -> streamlink 解析真实流
  -> ffmpeg 解码 PCM
  -> Silero VAD 切语音段
  -> Kotoba Whisper ASR
  -> filters 过滤无效文本
  -> OpenAI 兼容 LLM 翻译
  -> Socket.IO new_message 推送
  -> output/ 自动落盘
```

设置保存：

```text
Web /settings POST
  -> 只接受 DEFAULT_CONFIG 里的键
  -> 类型按默认值转换
  -> save_config() 原子写盘
  -> ASR/VAD 相关键变化时触发 reload_event
```

## 扩展点

- 换 ASR 模型：改 `config.json` 的 `asr.asr_model_id` 和 `asr.asr_model_dir`，并确保 `asr.py` 的加载参数适配该模型。
- 换 LLM 服务：改 `translation.llm_base_url`、`translation.llm_api_key`、`translation.llm_model`。本地 OpenAI 兼容端点可空 Key，远端通常需要 Key。
- 调翻译风格：优先改 `game_hint` 和 `prompt_extra`，不要把场景逻辑硬编码进 `translator.py`。
- 调断句：改 `vad` 分组里的阈值和时长参数；修改 `vad.py` 前先补对应纯逻辑或集成测试。
- 增加导出格式：从 `web.py` 的 `/export/<fmt>` 和 `storage.py` 入手，保持落盘失败不影响主流程。
