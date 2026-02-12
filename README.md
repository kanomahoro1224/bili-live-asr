# Live ASR & Translator (Qwen + DeepSeek)

这是一个专为 VTuber/直播设计的实时语音识别与翻译工具。它能够捕获直播流音频，使用 **Qwen-Audio (通义千问)** 进行高精度的实时语音转文字 (ASR)，并利用 **DeepSeek-V3** 进行上下文感知的自然语言翻译，最后通过 Bilibili 弹幕或 Web 界面实时展示。

## ✨ 主要特性

*   **实时流监听**: 直接解析 Bilibili 直播流音频，无需额外推流设备。
*   **高精度 ASR**: 集成阿里通义千问 `qwen-audio-turbo` / `qwen3-asr` 实时语音识别模型，支持日文、中文、英文。
*   **AI 智能翻译**: 使用 DeepSeek-V3 模型进行翻译，支持 **上下文记忆**，根据前文对话进行更准确的意译，尤其擅长处理口语、梗和省略句。
*   **智能断句与 VAD**: 内置 Silero VAD (语音活动检测) 或依赖 ASR 模型的语义断句，减少无效请求。
*   **Web 监控台**: 提供基于 Flask + Socket.IO 的 Web 界面，实时查看识别原文与翻译结果，并支持移动端访问。
*   **Bilibili 弹幕联动**: (可选) 自动将翻译结果发送到 Bilibili 直播间弹幕。
*   **安全防护**: 内置基础的 Web 安全防护，防止简单的扫描和攻击。

## 🛠️ 环境要求

*   Python 3.8+
*   FFmpeg (需添加到环境变量或放置在脚本同级目录)
*   Windows / Linux / macOS

## 📦 安装步骤

1.  **克隆或下载本项目**
2.  **安装 Python 依赖**

    ```bash
    pip install flask flask-socketio eventlet requests numpy streamlink dashscope onnxruntime
    ```

    *注意：如果是为了使用 Qwen ASR，请确保 `dashscope` 库是最新版本：*
    ```bash
    pip install --upgrade dashscope
    ```

3.  **配置 FFmpeg**
    *   下载 FFmpeg 并将 `ffmpeg.exe` 放置在项目根目录，或者确保其在系统 PATH 环境变量中。

4.  **准备模型文件 (可选)**
    *   如果启用本地 VAD 功能，需要下载 `Silero VAD_v4.onnx` 模型并放置在项目根目录。

## ⚙️ 配置说明

工具首次运行后会生成 `config.json` 配置文件。你也可以手动创建或直接修改文件。

```json
{
    "web_password": "admin",           // Web 管理界面密码
    "game_hint": "杂谈",               // 当前直播内容提示（用于辅助 AI 翻译上下文）
    "bili_room_url": "https://live.bilibili.com/123456", // 目标直播间 URL
    "bili_room_id": "123456",          // 目标直播间 Room ID (用于发送弹幕)
    "dashscope_api_key": "YOUR_KEY",   // [必填] 阿里云 Dashscope API Key (用于 Qwen ASR)
    "deepseek_key": "YOUR_KEY",        // [必填] DeepSeek API Key (用于翻译)
    "deepseek_model": "deepseek-chat", // 翻译模型选择
    "asr_language": "ja",              // 识别语言 (ja=日语)
    "use_vad": false                   // 是否启用本地 VAD (建议 False，直接用 Qwen 的断句)
}
```

### 获取 API Key
*   **Qwen ASR**: 前往 [阿里云百炼 (DashScope)](https://bailian.console.aliyun.com/) 申请并获取 API Key。
*   **DeepSeek**: 前往 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请 API Key。

## 🚀 运行方式

1.  **启动脚本**

    ```bash
    python live_asr.py
    ```

2.  **访问 Web 界面**
    *   打开浏览器访问 `http://localhost:5000`
    *   输入配置的 `web_password` (默认为 `admin`) 登录。

3.  **开始识别**
    *   在 Web 界面点击“开始监听”按钮。
    *   系统将自动连接直播流 -> 提取音频 -> ASR 识别 -> DeepSeek 翻译 -> 输出结果。

## ⚠️ 免责声明

*   本工具仅供学习和交流使用，不得用于任何非法用途。
*   使用 Qwen ASR 和 DeepSeek API 会产生相应的 token 费用，请自行关注账户余额。
*   请遵守 Bilibili 直播规范，不要发送违规弹幕。

## 📄 开源协议

MIT License
