# 贡献指南

本指南覆盖 tong 的环境搭建、开发约定与提交前检查。沟通、文档和新增注释默认使用中文。

## 目录

- [环境搭建](#环境搭建)
- [验证安装](#验证安装)
- [本地文件与模型](#本地文件与模型)
- [开发约定](#开发约定)
- [提交前检查](#提交前检查)
- [Commit 规范](#commit-规范)
- [分支与 PR 流程](#分支与-pr-流程)
- [重依赖与协作分工](#重依赖与协作分工)

## 环境搭建

依赖与运行统一走 [uv](https://docs.astral.sh/uv/)。

系统依赖：

- **uv**：`uv --version` 应有输出。
- **ffmpeg / ffprobe**：须在 PATH 中，`ffmpeg -version` 和 `ffprobe -version` 应有输出。项目不提交 `ffmpeg.exe`，uv 也不管理 ffmpeg 二进制。
- **Nvidia 显卡驱动**：本地 ASR 默认走 CUDA，`nvidia-smi` 应能看到显卡。没有 CUDA 时可以把 ASR 设备改为 CPU，但性能会明显下降。

同步依赖：

```powershell
uv sync
```

Windows 下 `torch` / `torchaudio` 通过 `pyproject.toml` 的 `pytorch-cu126` 索引安装。不要随手把它们改回 PyPI 默认源。

启动 Web 控制台：

```powershell
uv run python app.py
```

也可以直接运行：

```bat
run.bat
```

`run.bat` 会把 uv 缓存放到项目根目录的 `.uv-cache/`，避免大包占满系统盘。

## 验证安装

```powershell
ffmpeg -version
ffprobe -version
uv run python -m compileall -q src tests
uv run python -m pytest tests -q
```

如果需要确认 CUDA：

```powershell
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 本地文件与模型

以下文件或目录属于本机运行状态，不应提交：

- `config.json`
- `bilicookie.json`
- `models/`
- `.venv/`
- `.uv-cache/`
- `output/`
- `ffmpeg.exe` / `ffprobe.exe`

ASR 模型按一个模型一个文件夹存放：

```text
models/kotoba-whisper-v2.2/
```

默认 ASR 模型是 `kotoba-tech/kotoba-whisper-v2.2`。如果更换模型，需要同时确认 `config.json` 的 `asr_model_id` 和 `asr_model_dir`，并检查 `src/livetrans/asr.py` 的加载参数是否适配。

`config.json` 可能包含 API Key、Bilibili cookie、CSRF 等敏感信息。不要在日志、Issue、PR 描述或提交说明里复述完整密钥。

## 开发约定

项目按“纯逻辑在下、重依赖延迟导入、编排在上”的方向组织。新增代码请沿用现有分层：

- **纯逻辑模块**：`config`、`filters`、`llm`、`prompt_loader`、`storage` 等不要依赖 torch、网络或 Web 运行时，优先写可独立运行的单测。
- **重依赖延迟导入**：`torch`、`transformers`、`silero_vad` 等尽量放在实际加载 ASR/VAD 的函数或类里，不要让普通测试导入模块时就加载数 GB 依赖。
- **ffmpeg 作为外部工具**：统一从 PATH 查找，不要恢复为根目录 `ffmpeg.exe` 绑定，也不要把 ffmpeg 二进制放进 uv 依赖。
- **LLM 调用统一走 `llm.LLMClient`**：不要在 `translator.py`、`web.py` 或其他模块重复写 OpenAI 兼容 HTTP 请求和鉴权逻辑。
- **提示词放在 `src/livetrans/prompts/`**：可编辑提示词不要硬编码进业务代码；需要变量时通过 `prompt_loader.render_prompt()` 注入。
- **配置项同步更新**：新增配置时同时更新 `DEFAULT_CONFIG` 和 `CONFIG_SECTIONS`，保证运行时扁平 dict 和写盘分组 JSON 都可用。
- **Web 设置只保存白名单字段**：`/settings` 只接受 `DEFAULT_CONFIG` 中存在的键，新增设置项前先确认默认值和类型转换。
- **不要混改 VAD 与 ASR**：调整 VAD 阈值、切分策略、ASR 模型、LLM 提示词时尽量分开提交，方便回滚和定位问题。

更完整的架构说明见 [CLAUDE.md](CLAUDE.md)。

## 提交前检查

提交前至少运行：

```powershell
uv run python -m compileall -q src tests
uv run python -m pytest tests -q
git diff --check
```

当前测试集中在 `tests/`。不要直接跑全仓库 pytest 收集参考目录，除非明确要验证参考项目。

涉及 Web 页面改动时，还需要手动启动：

```powershell
uv run python app.py
```

检查设置保存、启动/停止、直播间切换、翻译列表滚动、弹幕发送等受影响流程。

## Commit 规范

使用中文 Conventional Commits：

```text
<类型>(<可选范围>): <简短描述>
```

常用类型：

- `feat`：新功能
- `fix`：修 bug
- `refactor`：重构
- `docs`：文档
- `test`：测试
- `chore`：维护杂务

示例：

```text
feat(asr): 增加远程 ASR 模式
fix(web): 修复翻译列表自动滚动
docs: 补充 ffmpeg 外部依赖说明
test(filters): 覆盖游戏播报过滤别名
```

## 分支与 PR 流程

- 从主分支切功能分支：`feature/<简述>`、`fix/<简述>`、`docs/<简述>` 等。
- PR 标题使用 Conventional Commits 风格。
- PR 描述写清楚改动动机、主要做法和验证命令。
- 不要把 `config.json`、`bilicookie.json`、模型文件、输出记录、ffmpeg 二进制提交进仓库。
- 如果改动涉及登录、cookie、弹幕发送或 LLM Key，描述里只写字段名和行为，不贴真实值。

## 重依赖与协作分工

GitHub CI 或其他无 GPU 环境通常无法完整验证本地 ASR 端到端流程。提交时按影响范围补充验证说明：

| 类别 | 模块 / 范围 | 验证方式 |
| --- | --- | --- |
| 纯逻辑 | `filters` / `config` / `llm` / `translator` 的纯函数、提示词渲染、文档 | `compileall` + `pytest tests` |
| Web 交互 | `web.py` / `templates/index.html` / 设置保存 / 主题 / 登录 | 本地启动 Web 控制台手测 |
| 实时流水线 | `pipeline` / `audio` / `stream` / VAD 切分 / ASR 加载 | 本地实际连直播流验证 |
| CUDA / 模型 | `asr.py` / `vad.py` / 模型下载与加载参数 | 说明显卡型号、模型目录、设备、运行结果 |

改动重依赖路径时，PR 描述里请附上本地实跑信息，例如：显卡型号、ASR 引擎、模型目录、运行时长、是否能稳定输出 ASR/TL、是否有阻塞或异常日志。
