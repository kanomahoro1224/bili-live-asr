import os
import sys
import time
import json
import wave
import struct
import math
import csv
import shutil
import tempfile
import subprocess
import threading
import collections
from typing import Dict, Any, List

import numpy as np
import requests
from flask import Flask, jsonify, request, render_template_string, session, redirect, url_for, send_from_directory
from flask_socketio import SocketIO

# 尝试导入核心推理库（仅保留必要的库）
try:
    import onnxruntime as ort
    from streamlink import Streamlink
    CORE_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"核心库缺失: {e}")
    CORE_LIBS_AVAILABLE = False

# 检查 Qwen ASR 相关库（延迟导入，避免模块级初始化）
try:
    # 使用安全的方式检查库是否可用
    import importlib.util
    
    # 检查 dashscope 包是否存在
    dashscope_spec = importlib.util.find_spec('dashscope')
    qwen_omni_spec = importlib.util.find_spec('dashscope.audio.qwen_omni')
    
    if dashscope_spec is not None and qwen_omni_spec is not None:
        QWEN_ASR_AVAILABLE = True
        print("Qwen ASR 库可用")
    else:
        QWEN_ASR_AVAILABLE = False
        print("Qwen ASR 库不完整")
        
except Exception as e:
    print(f"Qwen ASR 库检查失败: {e}")
    QWEN_ASR_AVAILABLE = False

# 检查 Qwen ASR 引擎是否可用
if not QWEN_ASR_AVAILABLE:
    print("错误：Qwen ASR 库不可用！")
    print("请安装 Qwen ASR 库：")
    print("  pip install --upgrade dashscope")
    sys.exit(1)

# ================ 基础环境配置 ================
current_dir = os.path.dirname(os.path.abspath(__file__))
VAD_MODEL_PATH = os.path.join(current_dir, "Silero VAD_v4.onnx")
FFMPEG_EXE = os.path.join(current_dir, "ffmpeg.exe")
CONFIG_FILE = os.path.join(current_dir, "config.json")
COOKIE_FILE = os.path.join(current_dir, "bilicookie.json")

# 默认配置
DEFAULT_CONFIG = {
    "web_password": "admin",
    "game_hint": "杂谈",
    "prompt_extra": "",
    "bili_room_url": "https://live.bilibili.com/000000",
    "bili_room_id": "000000",
    "bili_cookie": "",
    "bili_csrf": "",
    # ASR 引擎选择（默认使用 Qwen ASR）
    "asr_engine": "qwen_asr",             # 默认使用 "qwen_asr"
    "dashscope_api_key": "",              # Qwen ASR API Key
    "asr_language": "ja",                 # ASR 识别语言（ja=日语, zh=中文, en=英语）
    # VAD & 过滤参数
    "use_vad": False,               # 是否使用 VAD 语音检测（False=关闭VAD，完全依赖ASR断句）
    "vad_threshold": 0.4,           # VAD 触发阈值
    "min_silence_duration": 0.6,    # 断句静音保留时长
    "no_speech_threshold": 0.6,     # 非人声概率阈值 (大于此值视为BGM/噪音)
    "min_avg_logprob": -1.0,        # 平均置信度阈值 (小于此值视为幻觉)
    "banned_words": "視聴, 字幕, MBC, Music, music, BGM, VIDEO, WATCH, Subscribe", # 垃圾词
    "max_record_time": 15,
    # DeepSeek API
    "deepseek_key": "",
    "deepseek_model": "deepseek-chat",
    # 安全设置
    "log_security_events": True,   # 是否记录安全事件（恶意扫描等）
    # 翻译上下文设置
    "use_translation_context": True,  # 是否使用上下文翻译
    "context_window_size": 5,         # 上下文窗口大小（保留最近N句）
    "max_context_buffer": 20,         # 最大缓冲区大小
}

# ================ 全局状态 ================
app = Flask(__name__)
# ⚠️ 公网部署前请务必修改此密钥！建议使用随机生成的强密码
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_change_me_immediately')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

history_buffer: List[Dict[str, Any]] = []
history_lock = threading.Lock()  # 线程安全锁
log_buffer = collections.deque(maxlen=200) # 内存日志缓冲
config: Dict[str, Any] = DEFAULT_CONFIG.copy()

is_running = False 
current_ffmpeg_proc = None

# ASR 引擎相关全局变量
qwen_asr_conversation = None  # Qwen ASR 对话实例
qwen_asr_results_buffer = []  # Qwen ASR 结果缓冲区

# 翻译上下文缓冲区
translation_context_buffer = collections.deque(maxlen=20)  # 保留最近20句的上下文

# ================ 自动保存 ================
AUTO_SAVE_DIR = os.path.join(current_dir, "output")
os.makedirs(AUTO_SAVE_DIR, exist_ok=True)

def auto_save_record(orig: str, tran: str, ts: float):
    """自动追加保存一条记录到 CSV 和 JSON 文件"""
    try:
        date_str = time.strftime("%Y-%m-%d", time.localtime(ts))
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

        # 写 CSV
        csv_path = os.path.join(AUTO_SAVE_DIR, f"{date_str}.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["时间", "原文", "译文"])
            writer.writerow([time_str, orig, tran])

        # 写 JSON（追加到数组）
        json_path = os.path.join(AUTO_SAVE_DIR, f"{date_str}.json")
        records = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, ValueError):
                records = []
        records.append({"time": time_str, "ts": ts, "orig": orig, "tran": tran})
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log("Save", f"自动保存失败: {e}")

# ================ 安全防护 ================
login_attempts = {}  # IP -> 尝试次数
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_TIME = 300  # 5分钟
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
request_times = {}  # IP -> 请求时间列表（用于频率限制）
MAX_REQUESTS_PER_MINUTE = 60

# ================ 请求过滤中间件 ================
@app.before_request
def filter_malicious_requests():
    """过滤恶意请求和协议探测"""
    # 获取原始请求数据
    try:
        # 检查是否是非 HTTP 协议探测
        if request.environ.get('werkzeug.request'):
            raw_data = request.environ.get('werkzeug.request').data
            if raw_data:
                # 检测 TLS/SSL 握手 (以 \x16\x03 开头)
                if raw_data[:2] in [b'\x16\x03', b'\x16\x02']:
                    if config.get("log_security_events", False):
                        log("Security", f"阻止 TLS 握手探测: {request.remote_addr}")
                    return "This is an HTTP server, not HTTPS", 400
                
                # 检测 SSH 协议
                if raw_data.startswith(b'SSH-'):
                    if config.get("log_security_events", False):
                        log("Security", f"阻止 SSH 探测: {request.remote_addr}")
                    return "This is not an SSH server", 400
                
                # 检测 Oracle TNS 协议
                if b'DESCRIPTION=' in raw_data and b'CONNECT_DATA=' in raw_data:
                    if config.get("log_security_events", False):
                        log("Security", f"阻止 Oracle 数据库探测: {request.remote_addr}")
                    return "This is not a database server", 400
                
                # 检测 WebLogic T3 协议
                if raw_data.startswith(b't3 '):
                    if config.get("log_security_events", False):
                        log("Security", f"阻止 WebLogic T3 探测: {request.remote_addr}")
                    return "This is not a WebLogic server", 400
                
                # 检测 RDP 协议
                if b'Cookie: mstshash=' in raw_data:
                    if config.get("log_security_events", False):
                        log("Security", f"阻止 RDP 探测: {request.remote_addr}")
                    return "This is not an RDP server", 400
        
        # 检查 User-Agent（可选，阻止明显的扫描器）
        user_agent = request.headers.get('User-Agent', '').lower()
        suspicious_agents = ['masscan', 'nmap', 'nikto', 'sqlmap', 'scanner']
        if any(agent in user_agent for agent in suspicious_agents):
            if config.get("log_security_events", False):
                log("Security", f"阻止可疑 User-Agent: {request.remote_addr} - {user_agent}")
            return "Forbidden", 403
            
    except Exception as e:
        # 如果检查过程出错，记录但不阻止正常请求
        if config.get("log_security_events", False):
            log("Security", f"请求过滤检查出错: {e}")
    
    return None  # 允许请求继续

# ================ 日志工具 ================
def log(tag: str, msg: str):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    entry = f"[{timestamp}] [{tag}] {msg}"
    print(entry)
    log_buffer.append(entry)

# ================ 配置管理 ================
def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    if k not in saved: saved[k] = v
                config = saved
        except Exception as e: log("Config", f"配置文件加载失败: {e}")
    else: save_config()

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e: log("Config", f"配置文件保存失败: {e}")

load_config()

class QwenASRCallback:
    """Qwen ASR 识别回调处理器"""
    
    def __init__(self, conversation):
        self.conversation = conversation
        self.handlers = {
            'session.created': self._handle_session_created,
            'conversation.item.input_audio_transcription.completed': self._handle_final_text,
            'conversation.item.input_audio_transcription.text': self._handle_stash_text,
            'input_audio_buffer.speech_started': lambda r: None,  # 静默处理
            'input_audio_buffer.speech_stopped': lambda r: None   # 静默处理
        }
        
    def on_open(self):
        log("QwenASR", "连接已建立")
        
    def on_close(self, code, msg):
        log("QwenASR", f"连接已关闭 (code: {code})")
        
    def on_event(self, response):
        global qwen_asr_results_buffer
        try:
            handler = self.handlers.get(response.get('type'))
            if handler:
                handler(response)
        except Exception as e:
            log("Error", f"Qwen ASR 回调处理出错: {e}")
    
    def _handle_session_created(self, response):
        session_id = response.get('session', {}).get('id', 'unknown')
        log("QwenASR", f"会话已创建: {session_id}")
        
    def _handle_final_text(self, response):
        global qwen_asr_results_buffer
        try:
            text = response.get('transcript', '').strip()
            
            if text:
                result_item = {
                    'timestamp': time.time(),
                    'text': text,
                    'confidence': 1.0  # Qwen ASR 没有直接置信度
                }
                qwen_asr_results_buffer.append(result_item)
                log("QwenASR", f"收到完整句子: {text}")
                    
        except Exception as e:
            log("Error", f"Qwen ASR 最终结果处理出错: {e}")
    
    def _handle_stash_text(self, response):
        try:
            text = response.get('stash', '').strip()
            if text:
                # 只记录最后一次临时结果，避免日志刷屏
                pass  # 不输出临时识别结果
        except Exception as e:
            log("Error", f"Qwen ASR 临时结果处理出错: {e}")

def init_asr_engines():
    """初始化 ASR 引擎（仅支持 Qwen ASR）"""
    global qwen_asr_conversation
    
    asr_engine = config.get("asr_engine", "qwen_asr")
    log("Init", f"准备 ASR 引擎: {asr_engine}")
    
    if asr_engine == "qwen_asr" and QWEN_ASR_AVAILABLE:
        log("Init", "进入 Qwen ASR 分支")
        api_key = config.get("dashscope_api_key", "")
        log("Init", f"API Key 检查: {'已配置' if api_key else '未配置'}")
        
        if not api_key or api_key == "your_api_key_here":
            log("Error", "Qwen ASR 需要配置有效的 dashscope_api_key")
            log("Info", "请在 Web 界面的常规设置中配置 API Key")
            qwen_asr_conversation = None
        else:
            log("Init", "开始初始化 Qwen ASR...")
            # 直接在启动时初始化 Qwen ASR
            if init_qwen_asr_immediate():
                log("Init", "Qwen ASR 初始化成功")
            else:
                log("Error", "Qwen ASR 初始化失败")
                qwen_asr_conversation = None
        
    elif asr_engine == "qwen_asr" and not QWEN_ASR_AVAILABLE:
        log("Error", "Qwen ASR 库不可用，请升级 dashscope")
        qwen_asr_conversation = None
        
    else:
        log("Error", f"不支持的 ASR 引擎: {asr_engine}")
        log("Error", f"当前系统仅支持 Qwen ASR 引擎")
        qwen_asr_conversation = None

    log("Init", "init_asr_engines 函数完成")

def init_qwen_asr_immediate():
    """立即初始化 Qwen ASR（在主线程中直接初始化）"""
    global qwen_asr_conversation
    
    # 如果已经初始化且连接正常，直接返回
    if qwen_asr_conversation is not None:
        try:
            # 检查连接是否仍然有效
            return True
        except:
            # 连接已断开，需要重新初始化
            log("Init", "检测到 Qwen ASR 连接已断开，准备重新初始化")
            qwen_asr_conversation = None
    
    api_key = config.get("dashscope_api_key", "")
    if not api_key or api_key == "your_api_key_here":
        log("Error", "Qwen ASR API Key 未配置")
        return False
    
    try:
        log("Init", "正在初始化 Qwen ASR（主线程模式）...")
        
        # 直接在主线程中初始化
        from dashscope.audio.qwen_omni import OmniRealtimeConversation, OmniRealtimeCallback, MultiModality
        from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
        import dashscope
        
        dashscope.api_key = api_key
        
        # 创建回调类
        class MainThreadQwenASRCallback(OmniRealtimeCallback):
            def __init__(self, conversation):
                self.conversation = conversation
                self.handlers = {
                    'session.created': self._handle_session_created,
                    'conversation.item.input_audio_transcription.completed': self._handle_final_text,
                    'conversation.item.input_audio_transcription.text': self._handle_stash_text,
                    'input_audio_buffer.speech_started': lambda r: None,  # 静默处理
                    'input_audio_buffer.speech_stopped': lambda r: None   # 静默处理
                }
                
            def on_open(self):
                log("QwenASR", "连接已建立")
                
            def on_close(self, code, msg):
                global qwen_asr_conversation
                log("QwenASR", f"连接已关闭 (code: {code}, msg: {msg})")
                # 如果是超时断开，标记需要重连
                if msg and b'idle too long' in msg:
                    log("QwenASR", "检测到空闲超时，将在下次使用时自动重连")
                    qwen_asr_conversation = None
                
            def on_event(self, response):
                global qwen_asr_results_buffer
                try:
                    handler = self.handlers.get(response.get('type'))
                    if handler:
                        handler(response)
                except Exception as e:
                    log("Error", f"Qwen ASR 回调处理出错: {e}")
            
            def _handle_session_created(self, response):
                session_id = response.get('session', {}).get('id', 'unknown')
                log("QwenASR", f"会话已创建: {session_id}")
                
            def _handle_final_text(self, response):
                global qwen_asr_results_buffer
                try:
                    text = response.get('transcript', '').strip()
                    
                    if text:
                        result_item = {
                            'timestamp': time.time(),
                            'text': text,
                            'confidence': 1.0
                        }
                        qwen_asr_results_buffer.append(result_item)
                        log("QwenASR", f"收到完整句子: {text}")
                                
                except Exception as e:
                    log("Error", f"Qwen ASR 最终结果处理出错: {e}")
            
            def _handle_stash_text(self, response):
                try:
                    text = response.get('stash', '').strip()
                    if text:
                        # 只记录最后一次临时结果，避免日志刷屏
                        pass  # 不输出临时识别结果
                except Exception as e:
                    log("Error", f"Qwen ASR 临时结果处理出错: {e}")
        
        # 先创建临时回调
        temp_callback = MainThreadQwenASRCallback(None)
        
        log("Init", "创建 Qwen ASR 对话实例...")
        conversation = OmniRealtimeConversation(
            model='qwen3-asr-flash-realtime',
            url='wss://dashscope.aliyuncs.com/api-ws/v1/realtime',
            callback=temp_callback
        )
        
        # 设置回调的 conversation 引用
        temp_callback.conversation = conversation
        
        # 连接服务
        log("Init", "连接 Qwen ASR 服务...")
        conversation.connect()
        
        # 配置转录参数
        log("Init", "配置转录参数...")
        # 从配置中读取语言设置，默认为日语
        asr_language = config.get("asr_language", "ja")
        log("Init", f"ASR 识别语言: {asr_language}")
        
        transcription_params = TranscriptionParams(
            language=asr_language,  # 使用配置的语言
            sample_rate=16000,
            input_audio_format="pcm"
        )
        
        conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params
        )
        
        qwen_asr_conversation = conversation
        log("Init", "Qwen ASR 初始化完成（主线程）")
        return True
        
    except Exception as e:
        log("Error", f"Qwen ASR 初始化出错: {e}")
        import traceback
        traceback.print_exc()
        qwen_asr_conversation = None
        return False

def start_qwen_asr():
    """启动 Qwen ASR 识别服务（如果已经初始化则直接返回）"""
    global qwen_asr_conversation
    if qwen_asr_conversation:
        log("QwenASR", "识别服务已就绪")
        return True
    else:
        log("Error", "Qwen ASR 对话实例未初始化")
        return False

def stop_qwen_asr():
    """停止 Qwen ASR 识别服务"""
    global qwen_asr_conversation
    if qwen_asr_conversation:
        try:
            qwen_asr_conversation.close()
            log("QwenASR", "识别服务已停止")
        except Exception as e:
            log("Error", f"Qwen ASR 停止失败: {e}")

def send_audio_to_qwen_asr(audio_data: bytes):
    """发送音频数据到 Qwen ASR"""
    global qwen_asr_conversation
    
    if qwen_asr_conversation:
        try:
            # 将音频数据编码为 base64
            import base64
            audio_b64 = base64.b64encode(audio_data).decode('ascii')
            qwen_asr_conversation.append_audio(audio_b64)
        except Exception as e:
            log("Error", f"Qwen ASR 音频发送失败: {e}")
            # 如果发送失败，可能是连接已断开，标记需要重连
            if "closed" in str(e).lower() or "connection" in str(e).lower():
                log("QwenASR", "检测到连接异常，标记需要重连")
                qwen_asr_conversation = None
    else:
        log("Error", "Qwen ASR 对话实例未初始化，无法发送音频")

def get_qwen_asr_results() -> List[str]:
    """获取 Qwen ASR 识别结果"""
    global qwen_asr_results_buffer
    if not qwen_asr_results_buffer:
        return []
    
    results = [item['text'] for item in qwen_asr_results_buffer]
    qwen_asr_results_buffer.clear()
    return results

# ASR 引擎将在主程序中初始化

# ================ 核心功能函数 ================

def get_stream_url() -> str:
    room_url = config.get("bili_room_url")
    if not room_url or "你的房间号" in room_url: return ""
    try:
        session = Streamlink()
        streams = session.streams(room_url)
        if not streams: return ""
        stream = streams.get("best") or next(iter(streams.values()))
        return stream.to_url()
    except: return ""

def asr_audio_with_timestamps(path: str):
    """ASR 识别函数 - 仅支持 Qwen ASR"""
    asr_engine = config.get("asr_engine", "qwen_asr")
    
    if asr_engine == "qwen_asr":
        # Qwen ASR 是实时流式的，不需要文件模式
        # 直接从缓冲区获取结果
        return get_qwen_asr_results_filtered()
    else:
        log("Error", f"不支持的 ASR 引擎: {asr_engine}")
        log("Error", "当前系统仅支持 Qwen ASR 引擎")
        return ["无"]

def get_qwen_asr_results_filtered() -> List[str]:
    """获取 Qwen ASR 识别结果并应用过滤"""
    global qwen_asr_results_buffer
    if not qwen_asr_results_buffer:
        return ["无"]
    
    try:
        # 处理垃圾词过滤
        banned_str = config.get("banned_words", "")
        blacklist = [w.strip() for w in banned_str.replace("，", ",").split(",") if w.strip()]
        
        # 应用过滤规则
        valid_texts = []
        for item in qwen_asr_results_buffer:
            text = item['text'].strip()
            
            # 垃圾词过滤
            if any(bad_word.lower() in text.lower() for bad_word in blacklist): 
                continue
            
            # 基础格式清洗
            if len(text) < 2 and not text.isalnum(): 
                continue
            if text.startswith("(") or text.startswith("（") or text.startswith("【") or text.startswith("["): 
                continue

            if len(text) > 0: 
                valid_texts.append(text)
        
        # 清空缓冲区
        qwen_asr_results_buffer.clear()
        
        return valid_texts if valid_texts else ["无"]
        
    except Exception as e:
        log("Error", f"Qwen ASR 结果过滤出错: {e}")
        return ["无"]

def translate_batch(texts: List[str]) -> List[str]:
    """翻译函数 - 支持上下文翻译"""
    global translation_context_buffer
    
    # 如果识别结果是"无"，直接跳过翻译
    if not texts or texts == ["无"]: 
        return ["..."] * len(texts)

    api_key = config.get("deepseek_key") 
    if not api_key: return ["【未配置Key】"] * len(texts)
    
    # 检查是否启用上下文翻译
    use_context = config.get("use_translation_context", True)
    context_window = config.get("context_window_size", 5)
    
    # 构建上下文
    context_text = ""
    if use_context and len(translation_context_buffer) > 0:
        # 获取最近的N句作为上下文
        recent_context = list(translation_context_buffer)[-context_window:]
        context_lines = []
        for ctx in recent_context:
            context_lines.append(f"原文: {ctx['orig']}")
            context_lines.append(f"译文: {ctx['tran']}")
        context_text = "\n".join(context_lines)
    
    input_text = "\n".join(texts)
    headers = { "Authorization": f"Bearer {api_key}", "Content-Type": "application/json" }
    
    # 构建 prompt
    if use_context and context_text:
        prompt_content = (
            f"现在是一个vtb(鹿乃)在玩/或[{config.get('game_hint')}]。请将下面的日文句子逐行翻译成中文。"
            "要求：严格保持行数对应，一行日文对应一行中文，不要合并句子，口语化，去掉标点符号。\n"
            f"{config.get('prompt_extra')}\n\n"
            "【前文对话上下文】（用于理解语境，不要翻译这部分）：\n"
            f"{context_text}\n\n"
            "【待翻译内容】（只翻译这部分，根据上下文理解代词和省略的内容）：\n"
            f"{input_text}"
        )
    else:
        prompt_content = (
            f"现在是一个vtb(鹿乃)在玩/或[{config.get('game_hint')}]。请将下面的日文句子逐行翻译成中文。"
            "要求：严格保持行数对应，一行日文对应一行中文，不要合并句子，口语化，去掉标点符号。\n"
            f"{config.get('prompt_extra')}\n"
            "待翻译内容：\n"
            f"{input_text}"
        )
    
    payload = {
        "model": config.get("deepseek_model", "deepseek-chat"),
        "messages": [{
            "role": "user",
            "content": prompt_content
        }],
        "temperature": 0.3,
    }
    
    try:
        resp = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        clean_trans = []
        for line in content.split('\n'):
            line = line.strip()
            if len(line) > 2 and line[0].isdigit() and line[1] in ['.', '、']: line = line[2:].strip()
            if line: clean_trans.append(line)
        while len(clean_trans) < len(texts): clean_trans.append("...")
        
        # 将翻译结果添加到上下文缓冲区
        if use_context:
            for i, orig in enumerate(texts):
                tran = clean_trans[i] if i < len(clean_trans) else ""
                translation_context_buffer.append({
                    'orig': orig,
                    'tran': tran
                })
        
        return clean_trans
    except Exception as e:
        log("Error", f"翻译错误: {e}")
        return ["(失败)"] * len(texts)

def translate_and_emit_async(texts: List[str], timestamp: float):
    """异步翻译并发送结果（在后台线程执行）"""
    try:
        trans = translate_batch(texts)
        for i, orig in enumerate(texts):
            tran = trans[i] if i < len(trans) else ""
            ts = timestamp + i*0.01
            new_item = {"ts": ts, "orig": orig, "tran": tran}
            with history_lock:
                history_buffer.append(new_item)
                if len(history_buffer) > 50:
                    history_buffer[:] = history_buffer[-50:]
            socketio.emit('new_message', new_item)
            # 自动保存到本地文件
            auto_save_record(orig, tran, ts)
    except Exception as e:
        log("Error", f"异步翻译出错: {e}")

def get_bili_creds():
    cookie = config.get("bili_cookie", "")
    csrf = config.get("bili_csrf", "")
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                if isinstance(c_data, list):
                    parts = []
                    for item in c_data:
                        parts.append(f"{item['name']}={item['value']}")
                        if item['name'] == "bili_jct": csrf = item['value']
                    cookie = "; ".join(parts)
                elif isinstance(c_data, dict):
                    cookie = c_data.get("bili_cookie", cookie)
                    csrf = c_data.get("bili_csrf", csrf)
        except: pass
    return cookie, csrf

def send_danmu(msg: str) -> Dict[str, Any]:
    cookie, csrf = get_bili_creds()
    room_id = config.get("bili_room_id")
    if not (cookie and csrf and room_id): return {"ok": False, "error": "Cookie/RoomID缺失"}

    url = "https://api.live.bilibili.com/msg/send"
    payload = {
        "bubble": 0, "msg": msg, "color": 16777215, "mode": 1, "fontsize": 25,
        "rnd": int(time.time()), "roomid": int(room_id), "csrf": csrf, "csrf_token": csrf,
    }
    headers = { "Cookie": cookie, "Referer": config.get("bili_room_url"), "User-Agent": "Mozilla/5.0" }
    try:
        r = requests.post(url, data=payload, headers=headers, timeout=10)
        res = r.json()
        if res.get("code") == 0: return {"ok": True}
        else: return {"ok": False, "error": res.get("message")}
    except Exception as e: return {"ok": False, "error": str(e)}

def save_pcm_to_wav(pcm_data, output_path):
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_data)

# ================ 后台线程 (VAD + ASR 核心逻辑) ================

def worker_loop():
    global is_running, current_ffmpeg_proc, history_buffer
    temp_wav = os.path.join(tempfile.gettempdir(), "live_asr_temp.wav")
    
    log("Core", "工作线程启动")
    
    # 添加启动延迟，确保主线程完全启动
    time.sleep(0.5)
    log("Core", "工作线程初始化开始")
    
    try:
        # === Silero VAD 常量 ===
        SAMPLE_RATE = 16000
        CHUNK_SIZE = 512  # 512 samples @ 16k = 32ms
        PRE_ROLL_CHUNKS = 10 # 300ms Pre-roll
        log("Core", "VAD 常量定义完成")

        # 检查是否需要加载 VAD 模型
        use_vad = config.get("use_vad", False)
        vad_session = None
        
        if use_vad:
            # 检查 VAD 模型文件
            log("Core", f"检查 VAD 模型: {VAD_MODEL_PATH}")
            if not os.path.exists(VAD_MODEL_PATH):
                log("Error", f"找不到 VAD 模型: {VAD_MODEL_PATH}")
                log("Error", "VAD 模型缺失，工作线程退出")
                return
            
            log("Core", "VAD 模型文件存在，准备加载...")
            
            # 分步加载 VAD 模型，添加详细日志
            log("Init", "步骤1: 导入 signal 模块...")
            import signal
            log("Init", "步骤2: 定义超时处理器...")
            
            def timeout_handler(signum, frame):
                raise TimeoutError("VAD 模型加载超时")
            
            log("Init", "步骤3: 设置超时保护...")
            # 设置 30 秒超时（仅在非 Windows 系统）
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                log("Init", "超时保护已设置 (30秒)")
            else:
                log("Init", "Windows 系统，跳过超时保护")
            
            log("Init", "步骤4: 开始加载 VAD 模型...")
            vad_session = ort.InferenceSession(VAD_MODEL_PATH)
            log("Init", "步骤5: VAD 模型加载完成")
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # 取消超时
                log("Init", "超时保护已取消")
                
            log("Init", "VAD 模型加载成功")
        else:
            log("Core", "VAD 已禁用，将完全依赖 Qwen ASR 进行断句")
        
    except Exception as e:
        log("Error", f"VAD 加载失败: {e}")
        if use_vad:
            log("Error", "VAD 模型加载失败，工作线程退出")
            return
        else:
            log("Info", "VAD 已禁用，继续运行")

    # 获取 ASR 引擎类型和 VAD 设置
    asr_engine = config.get("asr_engine", "qwen_asr")
    use_vad = config.get("use_vad", False)
    log("Core", f"使用 ASR 引擎: {asr_engine}")
    log("Core", f"VAD 模式: {'启用' if use_vad else '禁用 (完全依赖ASR断句)'}")
    
    # 如果使用 Qwen ASR，在启动时已经初始化
    if asr_engine == "qwen_asr":
        log("Core", "Qwen ASR 模式：已在启动时初始化完成")
        # Qwen ASR 已经在启动时初始化，无需额外操作

    while True:
        if not is_running:
            # 如果停止运行，等待但不停止 ASR 服务（保持连接）
            time.sleep(1)
            continue

        stream_url = get_stream_url()
        if not stream_url:
            log("Core", "无法获取流，等待重试...")
            time.sleep(5)
            continue

        log("Core", "流已连接，开始监听...")
        if not os.path.exists(FFMPEG_EXE):
            log("Error", f"未找到 FFmpeg")
            time.sleep(5)
            continue

        # 如果使用 Qwen ASR，检查服务状态并尝试重新初始化
        if asr_engine == "qwen_asr":
            if not qwen_asr_conversation:
                log("Error", "Qwen ASR 未初始化，尝试重新初始化...")
                if init_qwen_asr_immediate():
                    log("Info", "Qwen ASR 重新初始化成功")
                else:
                    log("Error", "Qwen ASR 重新初始化失败，跳过此次流处理")
                    time.sleep(5)
                    continue

        # 启动 FFmpeg (S16LE, 16k, Mono)
        cmd = [FFMPEG_EXE, "-y", "-loglevel", "quiet", "-i", stream_url, 
               "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=CHUNK_SIZE * 2)
        current_ffmpeg_proc = process

        # === 音频处理模式选择 ===
        use_vad = config.get("use_vad", False)
        
        try:
            if use_vad:
                # === VAD 模式：使用 VAD 状态机 ===
                log("Core", "使用 VAD 模式进行音频处理")
                
                # VAD 状态机变量
                ring_buffer = collections.deque(maxlen=PRE_ROLL_CHUNKS)
                triggered = False
                voiced_frames = []
                silence_counter = 0
                
                # Silero LSTM 状态初始化
                h_state = np.zeros((2, 1, 64), dtype=np.float32)
                c_state = np.zeros((2, 1, 64), dtype=np.float32)

                while is_running and process.poll() is None:
                    # 使用非阻塞读取，避免卡死
                    try:
                        chunk_bytes = process.stdout.read(CHUNK_SIZE * 2)
                    except Exception as e:
                        log("Error", f"音频读取失败: {e}")
                        break
                    
                    if not chunk_bytes or len(chunk_bytes) != CHUNK_SIZE * 2:
                        log("Core", "音频流中断或数据不完整")
                        break
                    
                    # --- 1. VAD 推理 ---
                    audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    input_tensor = audio_float32[np.newaxis, :]

                    ort_inputs = {
                        'input': input_tensor,
                        'sr': np.array([16000], dtype=np.int64),
                        'h': h_state,
                        'c': c_state
                    }
                    ort_outs = vad_session.run(None, ort_inputs)
                    speech_prob = ort_outs[0][0][0]
                    h_state, c_state = ort_outs[1], ort_outs[2]
                    
                    # --- 2. 获取动态参数 ---
                    vad_thresh = float(config.get("vad_threshold", 0.4))
                    min_silence_sec = float(config.get("min_silence_duration", 0.6))
                    post_roll_chunks = int(min_silence_sec * SAMPLE_RATE / CHUNK_SIZE)
                    max_record_limit = int(config.get("max_record_time", 15))

                    current_is_speech = speech_prob >= vad_thresh
                    
                    # --- 3. Qwen ASR 实时处理 ---
                    if asr_engine == "qwen_asr":
                        try:
                            send_audio_to_qwen_asr(chunk_bytes)
                        except Exception as e:
                            log("Error", f"Qwen ASR 音频处理出错: {e}")
                    
                    # --- 4. VAD 状态机逻辑 ---
                    if not triggered:
                        ring_buffer.append(chunk_bytes)
                        if current_is_speech:
                            triggered = True
                            silence_counter = 0
                            voiced_frames.extend(ring_buffer)
                            voiced_frames.append(chunk_bytes)
                            ring_buffer.clear()
                            
                            # Qwen ASR 模式：开始收集结果
                            if asr_engine == "qwen_asr":
                                # 清空之前的结果缓冲区
                                get_qwen_asr_results()
                    else:
                        voiced_frames.append(chunk_bytes)
                        
                        if current_is_speech:
                            silence_counter = 0 
                        else:
                            silence_counter += 1
                        
                        current_duration = (len(voiced_frames) * CHUNK_SIZE) / SAMPLE_RATE

                        if (silence_counter >= post_roll_chunks) or (current_duration > max_record_limit):
                            triggered = False
                            
                            full_audio = b''.join(voiced_frames)
                            # 过滤太短的噪音
                            if len(full_audio) > SAMPLE_RATE * 0.5 * 2: 
                                
                                # Qwen ASR 模式：直接获取过滤后的结果
                                texts = asr_audio_with_timestamps("")  # 传空字符串，因为不需要文件
                                
                                if texts and texts != ["无"]:
                                    log("ASR", f"识别: {texts}")
                                    # 异步翻译，不阻塞主循环
                                    current_ts = time.time()
                                    socketio.start_background_task(translate_and_emit_async, texts, current_ts)
                            
                            voiced_frames = []
                            silence_counter = 0
                            ring_buffer.clear()

                    time.sleep(0.001)
            
            else:
                # === 无 VAD 模式：完全依赖 Qwen ASR 断句 ===
                log("Core", "使用无 VAD 模式，完全依赖 Qwen ASR 断句")
                
                # 结果收集定时器
                last_result_time = time.time()
                result_check_interval = 2.0  # 每2秒检查一次结果
                
                while is_running and process.poll() is None:
                    # 使用非阻塞读取，避免卡死
                    try:
                        chunk_bytes = process.stdout.read(CHUNK_SIZE * 2)
                    except Exception as e:
                        log("Error", f"音频读取失败: {e}")
                        break
                    
                    if not chunk_bytes or len(chunk_bytes) != CHUNK_SIZE * 2:
                        log("Core", "音频流中断或数据不完整")
                        break
                    
                    # --- 持续发送音频到 Qwen ASR ---
                    if asr_engine == "qwen_asr":
                        try:
                            send_audio_to_qwen_asr(chunk_bytes)
                        except Exception as e:
                            log("Error", f"Qwen ASR 音频处理出错: {e}")
                    
                    # --- 定时收集 Qwen ASR 结果 ---
                    current_time = time.time()
                    if current_time - last_result_time >= result_check_interval:
                        texts = asr_audio_with_timestamps("")
                        
                        if texts and texts != ["无"]:
                            log("ASR", f"识别: {texts}")
                            # 异步翻译，不阻塞主循环
                            current_ts = time.time()
                            socketio.start_background_task(translate_and_emit_async, texts, current_ts)
                        
                        last_result_time = current_time

                    time.sleep(0.001)
                    
        except Exception as e:
            log("Error", f"Processing Loop Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if current_ffmpeg_proc:
                try:
                    current_ffmpeg_proc.terminate()
                    current_ffmpeg_proc.wait(timeout=2)
                except:
                    try:
                        current_ffmpeg_proc.kill()
                    except:
                        pass
                current_ffmpeg_proc = None
            log("Core", "音频处理循环已退出，等待重连...")

# ================ Flask 路由 ================

@app.before_request
def require_login():
    # 安全检查：防止路径遍历攻击
    if '..' in request.path or request.path.startswith('//'):
        return jsonify({"ok": False, "error": "非法路径"}), 403
    
    # 请求频率限制（仅对POST请求）
    if request.method == 'POST':
        client_ip = request.remote_addr
        current_time = time.time()
        
        if client_ip not in request_times:
            request_times[client_ip] = []
        
        # 清理1分钟前的记录
        request_times[client_ip] = [t for t in request_times[client_ip] if current_time - t < 60]
        
        # 检查频率
        if len(request_times[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
            return jsonify({"ok": False, "error": "请求过于频繁，请稍后再试"}), 429
        
        request_times[client_ip].append(current_time)
    
    allowed = ['login', 'static', 'danmu_css']
    if request.endpoint not in allowed and not session.get('logged_in'):
        return redirect(url_for('login'))

@app.route('/danmu.css')
def danmu_css(): return send_from_directory(current_dir, 'danmu.css')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        client_ip = request.remote_addr
        current_time = time.time()
        
        # 检查是否被锁定
        if client_ip in login_attempts:
            attempts, lock_time = login_attempts[client_ip]
            if lock_time and current_time < lock_time:
                remaining = int(lock_time - current_time)
                return render_template_string(LOGIN_HTML, error=f"登录失败次数过多，请等待 {remaining} 秒后重试")
            elif lock_time and current_time >= lock_time:
                # 锁定时间已过，重置
                login_attempts[client_ip] = [0, None]
        
        password = request.form.get("password", "")
        correct_password = config.get("web_password", "1224")
        
        if password == correct_password:
            # 登录成功，清除记录
            if client_ip in login_attempts:
                del login_attempts[client_ip]
            session['logged_in'] = True
            session.permanent = True
            return redirect("/")
        else:
            # 登录失败，记录尝试次数
            if client_ip not in login_attempts:
                login_attempts[client_ip] = [0, None]
            
            login_attempts[client_ip][0] += 1
            attempts = login_attempts[client_ip][0]
            
            if attempts >= MAX_LOGIN_ATTEMPTS:
                login_attempts[client_ip][1] = current_time + LOGIN_LOCKOUT_TIME
                return render_template_string(LOGIN_HTML, error=f"登录失败次数过多，账户已锁定 {LOGIN_LOCKOUT_TIME // 60} 分钟")
            else:
                remaining = MAX_LOGIN_ATTEMPTS - attempts
                return render_template_string(LOGIN_HTML, error=f"密码错误，剩余尝试次数: {remaining}")
    
    return render_template_string(LOGIN_HTML)

def clean_unicode(text: str) -> str:
    """清理无效的 Unicode 代理对字符"""
    result = []
    for char in text:
        code = ord(char)
        # 过滤掉无效的代理对字符 (U+D800 到 U+DFFF)
        if 0xD800 <= code <= 0xDFFF:
            continue
        try:
            # 确保字符可以编码为 UTF-8
            char.encode('utf-8')
            result.append(char)
        except UnicodeEncodeError:
            # 如果无法编码，跳过该字符
            continue
    return ''.join(result)

@app.route("/")
def index(): 
    cleaned_html = clean_unicode(INDEX_HTML)
    return render_template_string(cleaned_html)

@app.route("/latest")
def latest(): 
    # 权限检查
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    with history_lock:
        return jsonify(list(history_buffer))

@app.route("/status")
def status(): 
    # 状态接口可以公开访问（用于前端显示）
    return jsonify({"running": is_running})

@app.route("/logs")
def get_logs(): 
    # 权限检查
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    return jsonify({"logs": list(log_buffer)})

@app.route("/export/<fmt>")
def export_file(fmt):
    """下载当天的 CSV 或 JSON 导出文件"""
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    date_str = request.args.get('date', time.strftime('%Y-%m-%d'))
    # 防止路径遍历
    if '..' in date_str or '/' in date_str or '\\' in date_str:
        return jsonify({"ok": False, "error": "非法日期"}), 400
    if fmt == 'csv':
        filepath = os.path.join(AUTO_SAVE_DIR, f"{date_str}.csv")
    elif fmt == 'json':
        filepath = os.path.join(AUTO_SAVE_DIR, f"{date_str}.json")
    else:
        return jsonify({"ok": False, "error": "不支持的格式"}), 400
    if not os.path.exists(filepath):
        return jsonify({"ok": False, "error": f"没有 {date_str} 的记录"}), 404
    return send_from_directory(AUTO_SAVE_DIR, os.path.basename(filepath), as_attachment=True)

@app.route("/toggle_run", methods=["POST"])
def toggle_run():
    # 权限检查
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    
    global is_running
    action = (request.json or {}).get("action")
    
    if action == "start": 
        is_running = True
    elif action == "stop": 
        is_running = False
    else:
        return jsonify({"ok": False, "error": "无效的操作"}), 400
    
    log("Ctrl", f"用户操作: {action}")
    socketio.emit('status_update', {"running": is_running})
    return jsonify({"running": is_running})

@app.route("/settings", methods=["GET", "POST"])
def settings_api():
    # 权限检查
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    
    if request.method == "GET": 
        return jsonify(config)
    
    new_data = request.json or {}
    
    # 验证和过滤输入
    allowed_keys = set(DEFAULT_CONFIG.keys())
    filtered_data = {}
    
    for k, v in new_data.items():
        if k in allowed_keys:
            # 类型验证
            default_value = DEFAULT_CONFIG[k]
            if isinstance(default_value, (int, float)):
                try:
                    filtered_data[k] = type(default_value)(v)
                except (ValueError, TypeError):
                    continue
            elif isinstance(default_value, str):
                # 字符串长度限制
                if isinstance(v, str) and len(v) < 10000:
                    filtered_data[k] = v
            else:
                filtered_data[k] = v
    
    # 检查是否需要重新初始化 ASR 引擎
    asr_engine_changed = False
    if 'asr_engine' in filtered_data and filtered_data['asr_engine'] != config.get('asr_engine'):
        asr_engine_changed = True
    if 'dashscope_api_key' in filtered_data and filtered_data['dashscope_api_key'] != config.get('dashscope_api_key'):
        asr_engine_changed = True
    if 'asr_language' in filtered_data and filtered_data['asr_language'] != config.get('asr_language'):
        asr_engine_changed = True
    if 'use_vad' in filtered_data and filtered_data['use_vad'] != config.get('use_vad'):
        asr_engine_changed = True  # VAD 模式改变需要重启工作线程
    
    # 更新配置
    for k, v in filtered_data.items():
        config[k] = v
    
    save_config()
    
    # 如果 ASR 引擎配置改变，重新初始化
    if asr_engine_changed:
        try:
            # 停止当前的 Qwen ASR 服务
            stop_qwen_asr()
            # 重新初始化 ASR 引擎（立即初始化）
            init_asr_engines()
            log("Config", "ASR 引擎配置已更新并重新初始化")
        except Exception as e:
            log("Error", f"ASR 引擎重新初始化失败: {e}")
    
    return jsonify({"ok": True})

@app.route("/send", methods=["POST"])
def send():
    # 权限检查
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    
    text = (request.json or {}).get("text", "")
    if not text or not text.strip(): 
        return jsonify({"ok": False, "error": "内容为空"})
    
    # 长度限制
    if len(text) > 200:
        return jsonify({"ok": False, "error": "内容过长，最大200字符"})
    
    return jsonify(send_danmu(text))

@app.route("/system/update", methods=["POST"])
def system_update():
    # 权限检查：必须已登录
    if not session.get('logged_in'):
        return jsonify({"ok": False, "error": "未授权"}), 403
    
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "未选择文件"})
    
    # 文件名安全检查
    filename = f.filename
    if not filename or not filename.endswith(".py"):
        return jsonify({"ok": False, "error": "仅支持 .py 文件"})
    
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({"ok": False, "error": "文件名非法"})
    
    # 文件大小检查
    f.seek(0, 2)  # 移动到文件末尾
    file_size = f.tell()
    f.seek(0)  # 重置到开头
    
    if file_size > MAX_UPLOAD_SIZE:
        return jsonify({"ok": False, "error": f"文件过大，最大 {MAX_UPLOAD_SIZE // 1024 // 1024}MB"})
    
    if file_size < 100:  # 至少100字节
        return jsonify({"ok": False, "error": "文件过小，可能无效"})
    
    # 读取文件内容进行基本验证
    try:
        content = f.read()
        f.seek(0)  # 重置以便后续保存
        
        # 基本Python语法检查：至少包含一些Python关键字
        if not any(keyword in content.decode('utf-8', errors='ignore') for keyword in ['import', 'def ', 'class ', '=']):
            return jsonify({"ok": False, "error": "文件内容验证失败，可能不是有效的Python文件"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"文件读取失败: {str(e)}"})
    
    # 确定目标文件路径（使用当前目录和明确的文件名）
    target_file = os.path.join(current_dir, "live_asr.py")
    backup_file = os.path.join(current_dir, f"live_asr_backup_{int(time.time())}.py")
    
    try:
        # 备份当前文件
        if os.path.exists(target_file):
            shutil.copy2(target_file, backup_file)
            log("Update", f"已备份当前文件到: {backup_file}")
        
        # 保存新文件
        f.save(target_file)
        log("Update", f"文件已更新: {target_file}")
        
        # 延迟重启
        def restart():
            time.sleep(2)
            try:
                os.execl(sys.executable, sys.executable, *sys.argv)
            except Exception as e:
                log("Error", f"重启失败: {e}")
        
        threading.Thread(target=restart, daemon=True).start()
        return jsonify({"ok": True, "message": "文件已更新，系统将在2秒后重启"})
        
    except Exception as e:
        log("Error", f"文件更新失败: {e}")
        # 如果保存失败，尝试恢复备份
        if os.path.exists(backup_file) and not os.path.exists(target_file):
            try:
                shutil.copy2(backup_file, target_file)
                log("Update", "已从备份恢复文件")
            except:
                pass
        return jsonify({"ok": False, "error": f"保存失败: {str(e)}"})

# ================ HTML 模板 ================

LOGIN_HTML = """
<!doctype html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>登录</title>
<style>body{display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);font-family:sans-serif}.card{background:rgba(255,255,255,0.9);padding:40px;border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,0.1);width:100%;max-width:340px;text-align:center}input{width:100%;padding:14px;margin-bottom:20px;border:2px solid #e5e7eb;border-radius:10px;box-sizing:border-box}button{width:100%;padding:14px;background:#3b82f6;color:white;border:none;border-radius:10px;cursor:pointer}.error{background:#fee2e2;color:#b91c1c;padding:10px;border-radius:8px;margin-bottom:20px}</style>
</head><body><div class="card"><h2>AI 同传控制台</h2>{% if error %}<div class="error">{{ error }}</div>{% endif %}<form method="POST"><input type="password" name="password" placeholder="输入密码" required autofocus><button type="submit">进入</button></form></div></body></html>
"""

INDEX_HTML = """
<!doctype html>
<html lang="zh">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI 同传控制台 (VAD Pro)</title>
    <link rel="stylesheet" href="/danmu.css">
    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <style>
        :root { --primary: #4f46e5; --bg: #f3f4f6; --text: #1f2937; --sidebar-bg: #f9fafb; --sidebar-hover: #e5e7eb; }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: -apple-system, sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }
        
        .navbar { height: 50px; background: #fff; border-bottom: 1px solid #e5e7eb; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; }
        .nav-brand { font-weight: 800; display: flex; align-items: center; gap: 10px; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #d1d5db; }
        .status-dot.active { background: #10b981; box-shadow: 0 0 5px #10b981; }
        .nav-btn { background: transparent; border: 1px solid #e5e7eb; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; color: #4b5563; display: flex; align-items: center; gap: 6px; }
        .nav-btn:hover { background: #f3f4f6; color: var(--primary); }

        .container { display: flex; padding: 15px; height: calc(100vh - 50px); width: 100%; }
        .player-wrapper { flex: 0 0 60%; background: #000; border-radius: 12px; overflow: hidden; min-width: 300px; max-width: 90%; display: flex; }
        .player-wrapper iframe { width: 100%; height: 100%; border: none; }
        .resizer { width: 12px; cursor: col-resize; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .resizer::after { content: ""; width: 4px; height: 40px; background: #d1d5db; border-radius: 2px; }

        .panel { flex: 1; background: #fff; border-radius: 12px; border: 1px solid #e5e7eb; display: flex; flex-direction: column; overflow: hidden; min-width: 300px; }
        .panel-header { padding: 12px 16px; border-bottom: 1px solid #f3f4f6; font-weight: 600; font-size: 14px; background: #fafafa; }
        .scroll-area { flex: 1; overflow-y: auto; padding: 15px; background: #f9fafb; scroll-behavior: smooth; }
        
        .msg-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
        .msg-header { font-size: 12px; color: #9ca3af; margin-bottom: 6px; }
        .edit-grid { display: grid; grid-template-columns: 1fr 1fr 40px; gap: 8px; }
        .edit-input { width: 100%; padding: 8px; border: 1px solid #e5e7eb; border-radius: 6px; font-size: 13px; resize: none; height: 48px; font-family: inherit; }
        .edit-input.orig { background: #f3f4f6; color: #4b5563; }
        .send-btn-mini { background: #eff6ff; color: var(--primary); border: 1px solid #c7d2fe; border-radius: 6px; cursor: pointer; display: flex; align-items: center; justify-content: center; }

        .chat-input-area { padding: 10px; background: #fff; border-top: 1px solid #e5e7eb; }
        .chat-box { display: flex; flex-direction: column; border: 1px solid #d1d5db; border-radius: 8px; padding: 8px; }
        .chat-textarea { border: none; outline: none; width: 100%; resize: none; height: 40px; font-size: 14px; margin-bottom: 5px; }
        .chat-tools { display: flex; justify-content: space-between; align-items: center; }
        .chat-send-btn { background: #00aeec; color: white; border: none; padding: 6px 16px; border-radius: 4px; cursor: pointer; }

        /* === 新版设置面板样式 === */
        .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.4); backdrop-filter: blur(4px); z-index: 200; align-items: center; justify-content: center; }
        .modal-box { background: #fff; width: 90%; max-width: 850px; height: 80vh; border-radius: 12px; display: flex; overflow: hidden; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); position: relative; }
        
        /* 侧边栏 */
        .sidebar { width: 180px; background: var(--sidebar-bg); border-right: 1px solid #e5e7eb; display: flex; flex-direction: column; padding: 15px 0; }
        .sidebar-title { font-weight: 800; padding: 0 20px 15px; font-size: 16px; color: #374151; border-bottom: 1px solid #e5e7eb; margin-bottom: 10px; }
        .tab-btn { padding: 12px 20px; cursor: pointer; color: #4b5563; font-size: 14px; font-weight: 500; transition: all 0.2s; border-left: 3px solid transparent; display: flex; align-items: center; gap: 8px; }
        .tab-btn:hover { background: var(--sidebar-hover); color: #111; }
        .tab-btn.active { background: #fff; border-left-color: var(--primary); color: var(--primary); font-weight: 600; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        
        /* 内容区 */
        .content-area { flex: 1; display: flex; flex-direction: column; background: #fff; overflow: hidden; }
        .content-header { padding: 15px 25px; border-bottom: 1px solid #f3f4f6; display: flex; justify-content: space-between; align-items: center; }
        .content-title { font-size: 18px; font-weight: 700; color: #1f2937; }
        .close-btn { cursor: pointer; font-size: 20px; color: #9ca3af; padding: 5px; }
        .close-btn:hover { color: #4b5563; }
        
        .tab-content { display: none; padding: 25px; overflow-y: auto; flex: 1; animation: fadeIn 0.2s; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }

        /* 表单元素 */
        .form-group { margin-bottom: 20px; }
        .form-label { display: block; font-size: 13px; font-weight: 700; color: #4b5563; margin-bottom: 8px; }
        .form-input { width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px; transition: border 0.2s; }
        .form-input:focus { border-color: var(--primary); outline: none; box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1); }
        .form-helper { font-size: 12px; color: #6b7280; margin-top: 5px; line-height: 1.4; }
        
        /* 标签式输入框样式 */
        .input-group { display: flex; gap: 10px; margin-bottom: 10px; }
        .btn-add { background: #374151; color: white; border: none; padding: 0 20px; border-radius: 6px; cursor: pointer; font-size: 13px; transition: background 0.2s; white-space: nowrap; }
        .btn-add:hover { background: #1f2937; }
        .tag-container { display: flex; flex-wrap: wrap; gap: 8px; min-height: 40px; align-content: flex-start; }
        .tag-item { background: #e5e7eb; color: #374151; padding: 6px 12px; border-radius: 4px; font-size: 13px; font-weight: 500; display: inline-flex; align-items: center; gap: 8px; transition: all 0.2s; }
        .tag-item:hover { background: #d1d5db; }
        .tag-close { cursor: pointer; color: #9ca3af; font-weight: bold; font-size: 14px; line-height: 1; }
        .tag-close:hover { color: #ef4444; }
        .tag-count { font-size: 12px; color: #9ca3af; margin-bottom: 5px; }
        
        /* 底部按钮 */
        .modal-footer { padding: 15px 25px; border-top: 1px solid #f3f4f6; text-align: right; background: #fff; }
        .save-btn { padding: 10px 24px; background: var(--primary); color: white; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; }
        .save-btn:hover { background: #4338ca; }

        /* 日志控制台 */
        #logConsole { background: #1e1e1e; color: #10b981; font-family: monospace; padding: 15px; border-radius: 8px; height: 100%; overflow-y: scroll; font-size: 12px; white-space: pre-wrap; }

        @media (max-width: 800px) { 
            .container { flex-direction: column; } 
            .modal-box { flex-direction: column; height: 90vh; }
            .sidebar { width: 100%; flex-direction: row; overflow-x: auto; border-right: none; border-bottom: 1px solid #e5e7eb; padding: 0; }
            .sidebar-title { display: none; }
            .tab-btn { flex: 1; justify-content: center; padding: 15px; border-left: none; border-bottom: 3px solid transparent; }
            .tab-btn.active { border-left: none; border-bottom-color: var(--primary); }
        }
    </style>
</head>
<body>
    <div class="navbar">
        <div class="nav-brand"><span>同传姬 (VAD Pro)</span><div id="statusDot" class="status-dot"></div></div>
        <div style="display:flex;gap:8px;align-items:center">
            <button class="nav-btn" onclick="exportData('csv')"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> CSV</button>
            <button class="nav-btn" onclick="exportData('json')"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> JSON</button>
            <button class="nav-btn" onclick="openModal()"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg> 控制台</button>
        </div>
    </div>

    <div class="container" id="container">
        <div class="player-wrapper" id="leftPanel">
            <iframe id="liveFrame" src="https://www.bilibili.com/blackboard/live/live-activity-player.html?cid=15152878&logo=0&mute=0" allowfullscreen></iframe>
        </div>
        <div class="resizer" id="dragMe"></div>
        <div class="panel">
            <div class="panel-header" style="display:flex;align-items:center;gap:6px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> 实时转写</div>
            <div class="scroll-area" id="transBox"></div>
            <div class="chat-input-area">
                <div class="chat-box">
                    <textarea id="chatInput" class="chat-textarea" placeholder="手动发送弹幕..." maxlength="40" oninput="document.getElementById('charCount').innerText=this.value.length+'/40'" onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat();}"></textarea>
                    <div class="chat-tools">
                        <span style="font-size:12px;color:#999" id="charCount">0/40</span>
                        <button class="chat-send-btn" onclick="sendChat()">发送</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- 设置弹窗 -->
    <div id="settingsModal" class="modal" onclick="if(event.target===this) closeModal()">
        <div class="modal-box">
            <!-- 侧边栏 -->
            <div class="sidebar">
                <div class="sidebar-title">设置菜单</div>
                <div class="tab-btn active" onclick="switchTab('general', this)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg> 常规设置</div>
                <div class="tab-btn" onclick="switchTab('vad', this)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg> VAD 参数</div>
                <div class="tab-btn" onclick="switchTab('filter', this)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg> 过滤设置</div>
                <div class="tab-btn" onclick="switchTab('logs', this)"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 7 4 4 20 4 20 7"></polyline><line x1="9" y1="20" x2="15" y2="20"></line><line x1="12" y1="4" x2="12" y2="20"></line></svg> 运行日志</div>
            </div>
            
            <!-- 内容区 -->
            <div class="content-area">
                <div class="content-header">
                    <div class="content-title" id="pageTitle">常规设置</div>
                    <div class="close-btn" onclick="closeModal()">✕</div>
                </div>

                <!-- 1. 常规设置 -->
                <div id="tab-general" class="tab-content active">
                    <button onclick="toggleRun()" id="runBtn" style="width:100%; padding:14px; background:#10b981; color:white; border:none; border-radius:8px; font-weight:bold; margin-bottom:25px; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:8px;"><svg id="runIcon" width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> <span>启动同传</span></button>
                    
                    <!-- ASR 引擎选择 -->
                    <div style="background:#f8fafc; padding:15px; border-radius:8px; margin-bottom:20px; border:1px solid #e2e8f0;">
                        <div style="font-weight:bold; color:#1e293b; margin-bottom:10px; font-size:14px; display:flex; align-items:center; gap:6px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg> ASR 引擎选择</div>
                        <div class="form-group">
                            <label class="form-label">识别引擎</label>
                            <select id="set_asr_engine" class="form-input" onchange="toggleAsrSettings()">
                                <option value="qwen_asr">Qwen ASR (云端API)</option>
                            </select>
                            <div class="form-helper">使用阿里云 DashScope 提供的实时语音识别服务</div>
                        </div>
                        <div id="qwen_asr_settings" style="display:none;">
                            <div class="form-group">
                                <label class="form-label">DashScope API Key</label>
                                <input type="password" id="set_dashscope_api_key" class="form-input" placeholder="sk-..." />
                                <div class="form-helper">获取地址: <a href="https://dashscope.console.aliyun.com/" target="_blank">https://dashscope.console.aliyun.com/</a></div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">识别语言</label>
                                <select id="set_asr_language" class="form-input">
                                    <option value="ja">日语 (Japanese)</option>
                                    <option value="zh">中文 (Chinese)</option>
                                    <option value="en">英语 (English)</option>
                                </select>
                                <div class="form-helper">选择要识别的语言，系统将专门识别该语言</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">游戏/场景名</label>
                        <input id="set_game_hint" class="form-input" placeholder="例如：杂谈、APEX" />
                    </div>
                    <div class="form-group">
                        <label class="form-label">额外 Prompt</label>
                        <textarea id="set_prompt_extra" class="form-input" style="height:60px;resize:vertical;" placeholder="给翻译模型的额外指令"></textarea>
                    </div>
                    
                    <!-- 上下文翻译设置 -->
                    <div style="background:#f0fdf4; padding:15px; border-radius:8px; margin-bottom:20px; border:1px solid #bbf7d0;">
                        <div style="font-weight:bold; color:#15803d; margin-bottom:10px; font-size:14px; display:flex; align-items:center; gap:6px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg> 上下文翻译</div>
                        <div class="form-group">
                            <label class="form-label">启用上下文翻译</label>
                            <select id="set_use_translation_context" class="form-input">
                                <option value="true">启用 - 根据前文理解代词和省略内容 (推荐)</option>
                                <option value="false">禁用 - 每句独立翻译</option>
                            </select>
                            <div class="form-helper">启用后，翻译时会参考前面几句对话，更准确理解"他"、"她"、"这个"等代词的含义</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">上下文窗口大小</label>
                            <input type="number" id="set_context_window_size" class="form-input" min="1" max="20" placeholder="5" />
                            <div class="form-helper">保留最近N句作为上下文（1-20句，推荐5句）。越大越准确但API调用成本越高</div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">直播间 URL</label>
                        <input id="set_bili_room_url" class="form-input" />
                    </div>
                    <div class="form-group">
                        <label class="form-label">直播间 Room ID (用于弹幕)</label>
                        <input id="set_bili_room_id" class="form-input" />
                    </div>
                    
                    <div style="border-top:1px solid #eee; margin-top:20px; padding-top:20px;">
                         <div style="border:2px dashed #e5e7eb; padding:15px; text-align:center; cursor:pointer; border-radius:8px; color:#6b7280; font-size:13px; display:flex; align-items:center; justify-content:center; gap:8px;" onclick="document.getElementById('cFile').click()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg> 
                            <span id="uTxt">点击上传新版代码 (.py)</span>
                            <input type="file" id="cFile" hidden accept=".py" onchange="handleFile(this)">
                        </div>
                        <button onclick="upCode()" style="width:100%;margin-top:10px;padding:8px;background:#4b5563;color:white;border:none;border-radius:6px;cursor:pointer;">更新并重启</button>
                    </div>
                </div>

                <!-- 2. VAD 设置 -->
                <div id="tab-vad" class="tab-content">
                    <div class="form-group">
                        <label class="form-label">VAD 语音检测</label>
                        <select id="set_use_vad" class="form-input" onchange="toggleVadSettings()">
                            <option value="false">禁用 - 完全依赖 ASR 断句 (推荐)</option>
                            <option value="true">启用 - 使用 VAD + ASR 双重断句</option>
                        </select>
                        <div class="form-helper">禁用 VAD 可以避免漏录语音，让 Qwen ASR 完全负责断句</div>
                    </div>
                    
                    <div id="vad_advanced_settings" style="display:none;">
                        <div class="form-group">
                            <label class="form-label">触发阈值 (Vad Threshold)</label>
                            <input type="number" id="set_vad_threshold" class="form-input" step="0.1" />
                            <div class="form-helper">范围 0.1-0.9。越小越灵敏，越大越难触发。</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">断句静音保留 (Min Silence Duration)</label>
                            <input type="number" id="set_min_silence" class="form-input" step="0.1" placeholder="0.6" />
                            <div class="form-helper">说话后静音多少秒才切断。调大(0.5-0.8)可保留尾音。</div>
                        </div>
                    </div>
                    
                    <div style="background:#f0f9ff; padding:15px; border-radius:8px; margin-top:10px;">
                        <div style="font-weight:bold; color:#0369a1; margin-bottom:10px; font-size:13px;">高级过滤参数 (在此处配置)</div>
                        <div class="form-group">
                            <label class="form-label">非人声概率阈值 (No Speech Prob)</label>
                            <input type="number" id="set_no_speech" class="form-input" step="0.1" placeholder="0.6" />
                            <div class="form-helper">大于此值(0.0-1.0)的片段会被丢弃。用于过滤纯BGM或噪音。推荐 0.6。</div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">最低置信度阈值 (Avg Logprob)</label>
                            <input type="number" id="set_min_logprob" class="form-input" step="0.1" placeholder="-1.0" />
                            <div class="form-helper">小于此值(通常是负数)会被丢弃。用于过滤模型的幻觉。推荐 -1.0。</div>
                        </div>
                    </div>
                </div>

                <!-- 3. 过滤设置 (标签式) -->
                <div id="tab-filter" class="tab-content">
                    <div class="form-group">
                        <label class="form-label">添加屏蔽词</label>
                        <div class="input-group">
                            <input id="new_ban_word" class="form-input" placeholder="请输入您要屏蔽的内容 (回车添加)" onkeydown="if(event.keyCode==13) addBanWord()">
                            <button class="btn-add" onclick="addBanWord()">添加</button>
                        </div>
                        <div class="tag-count">屏蔽列表 (<span id="tagCount">0</span>)</div>
                        <div id="ban_tag_list" class="tag-container"></div>
                        <input type="hidden" id="set_banned_words">
                        <div class="form-helper" style="margin-top:15px;">
                            提示：识别结果中如果包含这些词（如："Music", "字幕"），该句将直接丢弃。不区分大小写。
                        </div>
                    </div>
                </div>

                <!-- 4. 日志 -->
                <div id="tab-logs" class="tab-content">
                    <div id="logConsole">Connecting to log stream...</div>
                </div>

                <div class="modal-footer">
                    <button class="save-btn" onclick="saveSettings()">保存配置</button>
                </div>
            </div>
        </div>
    </div>

    <script>
    const socket = io();
    let logInterval = null;
    let bannedTags = [];

    // 拖拽逻辑
    const resizer = document.getElementById('dragMe'), left = document.getElementById('leftPanel'), con = document.getElementById('container'), iframe = document.getElementById('liveFrame');
    let x=0, w=0;
    resizer.onmousedown = e => { x=e.clientX; w=left.getBoundingClientRect().width; iframe.style.pointerEvents='none'; document.onmousemove=mm; document.onmouseup=mu; };
    function mm(e){ const nw = ((w + e.clientX - x)*100)/con.getBoundingClientRect().width; if(nw>20&&nw<80) left.style.flex=`0 0 ${nw}%`; }
    function mu(){ iframe.style.pointerEvents='auto'; document.onmousemove=null; }

    // Socket 逻辑
    socket.on('connect', ()=>document.getElementById('statusDot').classList.add('active'));
    socket.on('new_message', d => {
        const b = document.getElementById('transBox');
        const div = document.createElement('div'); div.className = 'msg-card';
        const now = new Date().toLocaleTimeString('zh-CN', {hour12: false});
        div.innerHTML = `<div class="msg-header">${now}</div>
            <div class="edit-grid"><textarea class="edit-input orig" readonly>${esc(d.orig)}</textarea>
            <textarea class="edit-input tran">${esc(d.tran)}</textarea><div class="send-btn-mini" onclick="sendTran(this)">➤</div></div>`;
        b.appendChild(div);
        b.scrollTop = b.scrollHeight;
    });
    socket.on('status_update', s => setRunBtn(s.running));
    
    // 初始化
    window.onload = async () => {
        try {
            setRunBtn((await (await fetch('/status')).json()).running);
            const h = await (await fetch('/latest')).json();
            const b = document.getElementById('transBox');
            h.forEach(d => {
                const div = document.createElement('div'); div.className = 'msg-card';
                div.innerHTML = `<div class="msg-header">${new Date(d.ts*1000).toLocaleTimeString()}</div>
            <div class="edit-grid"><textarea class="edit-input orig" readonly>${esc(d.orig)}</textarea>
            <textarea class="edit-input tran">${esc(d.tran)}</textarea><div class="send-btn-mini" onclick="sendTran(this)">➤</div></div>`;
                b.appendChild(div);
            });
            b.scrollTop = b.scrollHeight;
        } catch(e){}
    };

    function setRunBtn(r) { 
        const b=document.getElementById('runBtn');
        const span=b.querySelector('span');
        const icon=document.getElementById('runIcon');
        span.innerText=r?"停止同传":"启动同传";
        icon.innerHTML=r?'<rect x="6" y="6" width="12" height="12"></rect>':'<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        b.style.background=r?"#ef4444":"#10b981"; 
    }
    async function toggleRun() { const a = document.getElementById('runBtn').innerText.includes('启动')?'start':'stop'; await fetch('/toggle_run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:a})}); }
    
    // === Tab 切换逻辑 ===
    function switchTab(tabId, btn) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('pageTitle').innerText = btn.innerText.replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g,''); 
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById('tab-'+tabId).classList.add('active');
        if(tabId === 'logs') { if(!logInterval) { fetchLogs(); logInterval = setInterval(fetchLogs, 1500); } } 
        else { if(logInterval) { clearInterval(logInterval); logInterval = null; } }
    }

    // === 标签逻辑 ===
    function renderTags() {
        const c = document.getElementById('ban_tag_list'); c.innerHTML='';
        bannedTags.forEach((w,i)=>{ if(!w.trim())return; 
            const t=document.createElement('div'); t.className='tag-item';
            t.innerHTML=`<span>${esc(w)}</span><span class="tag-close" onclick="removeBanWord(${i})">×</span>`;
            c.appendChild(t);
        });
        document.getElementById('tagCount').innerText=bannedTags.length;
    }
    function addBanWord(){
        const i=document.getElementById('new_ban_word'); const v=i.value.trim();
        if(v && !bannedTags.includes(v)) { bannedTags.push(v); renderTags(); }
        i.value='';
    }
    function removeBanWord(i){ bannedTags.splice(i,1); renderTags(); }

    // === ASR 引擎切换逻辑 ===
    function toggleAsrSettings() {
        // 由于只支持 Qwen ASR，始终显示 Qwen ASR 设置
        const qwenAsrSettings = document.getElementById('qwen_asr_settings');
        qwenAsrSettings.style.display = 'block';
    }
    
    // === VAD 设置切换逻辑 ===
    function toggleVadSettings() {
        const useVad = document.getElementById('set_use_vad').value === 'true';
        const vadAdvancedSettings = document.getElementById('vad_advanced_settings');
        vadAdvancedSettings.style.display = useVad ? 'block' : 'none';
    }

    // === 设置存取逻辑 ===
    function openModal() {
        document.getElementById('settingsModal').style.display = 'flex';
        fetch('/settings').then(r=>r.json()).then(c => {
            // ASR 引擎设置（默认 Qwen ASR）
            document.getElementById('set_asr_engine').value = c.asr_engine||'qwen_asr';
            document.getElementById('set_dashscope_api_key').value = c.dashscope_api_key||'';
            document.getElementById('set_asr_language').value = c.asr_language||'ja';
            toggleAsrSettings(); // 显示 Qwen ASR 设置
            
            // 其他设置
            document.getElementById('set_game_hint').value = c.game_hint||'';
            document.getElementById('set_prompt_extra').value = c.prompt_extra||'';
            
            // 上下文翻译设置
            document.getElementById('set_use_translation_context').value = c.use_translation_context !== false ? 'true' : 'false';
            document.getElementById('set_context_window_size').value = c.context_window_size||5;
            
            document.getElementById('set_bili_room_url').value = c.bili_room_url||'';
            document.getElementById('set_bili_room_id').value = c.bili_room_id||'';
            document.getElementById('set_use_vad').value = c.use_vad ? 'true' : 'false';
            document.getElementById('set_vad_threshold').value = c.vad_threshold||0.4;
            document.getElementById('set_min_silence').value = c.min_silence_duration||0.6;
            document.getElementById('set_no_speech').value = c.no_speech_threshold||0.6;
            toggleVadSettings(); // 根据 VAD 设置显示/隐藏高级选项 
            document.getElementById('set_min_logprob').value = c.min_avg_logprob||-1.0; 
            
            const raw = c.banned_words || "";
            bannedTags = raw.replace(/，/g, ',').split(',').map(s => s.trim()).filter(s => s);
            renderTags();
        });
        switchTab('general', document.querySelector('.tab-btn')); 
    }
    
    function closeModal() { document.getElementById('settingsModal').style.display='none'; if(logInterval) clearInterval(logInterval); }
    
    async function saveSettings() {
        await fetch('/settings', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
            // ASR 引擎设置
            asr_engine: document.getElementById('set_asr_engine').value,
            dashscope_api_key: document.getElementById('set_dashscope_api_key').value,
            asr_language: document.getElementById('set_asr_language').value,
            
            // 其他设置
            game_hint: document.getElementById('set_game_hint').value,
            prompt_extra: document.getElementById('set_prompt_extra').value,
            
            // 上下文翻译设置
            use_translation_context: document.getElementById('set_use_translation_context').value === 'true',
            context_window_size: parseInt(document.getElementById('set_context_window_size').value) || 5,
            
            bili_room_url: document.getElementById('set_bili_room_url').value,
            bili_room_id: document.getElementById('set_bili_room_id').value,
            use_vad: document.getElementById('set_use_vad').value === 'true',
            vad_threshold: parseFloat(document.getElementById('set_vad_threshold').value),
            min_silence_duration: parseFloat(document.getElementById('set_min_silence').value),
            no_speech_threshold: parseFloat(document.getElementById('set_no_speech').value), 
            min_avg_logprob: parseFloat(document.getElementById('set_min_logprob').value),   
            banned_words: bannedTags.join(', ') 
        })});
        alert('配置已保存，如果切换了 ASR 引擎，建议重启服务');
        closeModal();
    }

    async function fetchLogs() { try { const d = await (await fetch('/logs')).json(); const el = document.getElementById('logConsole'); if(el.innerText.length!==d.logs.join('\\n').length){el.innerText=d.logs.join('\\n');el.scrollTop=el.scrollHeight;}} catch{} }
    async function sendChat() { const i=document.getElementById('chatInput'); if(!i.value.trim())return; await fetch('/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:i.value})}); i.value=''; document.getElementById('charCount').innerText='0/40'; }
    async function sendTran(btn) { const t=btn.parentElement.querySelector('.tran').value; if(!t)return; btn.innerText='...'; const r=await fetch('/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'['+t+']'})}); btn.innerText=(await r.json()).ok?'✔':'X'; setTimeout(()=>btn.innerText='➤',2000); }
    function exportData(fmt) { window.open('/export/' + fmt, '_blank'); }
    function handleFile(i){ if(i.files[0]) document.getElementById('uTxt').innerText=i.files[0].name; }
    async function upCode(){ 
        const f=document.getElementById('cFile').files[0]; 
        if(!f) return alert('请先选择文件'); 
        if(f.size > 5*1024*1024) return alert('文件过大，最大5MB'); 
        if(!confirm('确定要更新并重启系统吗？')) return; 
        try {
            const fd=new FormData(); 
            fd.append('file',f); 
            const res = await fetch('/system/update',{method:'POST',body:fd});
            const data = await res.json();
            if(data.ok) {
                alert('文件已更新，系统将在2秒后重启...');
            } else {
                alert('更新失败: ' + (data.error || '未知错误'));
            }
        } catch(e) {
            alert('上传失败: ' + e.message);
        }
    }
    function esc(s){ return s?s.replace(/&/g,'&amp;').replace(/</g,'&lt;'):''; }
    </script>
</body>
</html>
"""

def find_available_port(preferred=5231):
    """检查端口是否可用，不可用则随机分配一个"""
    import socket as _socket
    # 先尝试首选端口
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", preferred))
            return preferred
        except OSError:
            pass
    # 首选端口被占用，让系统分配随机端口
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    try:
        log("Init", "=== AI 同传系统启动 ===")
        
        log("Init", "步骤1: 检查必要文件...")
        # 检查必要文件
        if not os.path.exists(FFMPEG_EXE):
            log("Error", f"错误：未找到 ffmpeg - {FFMPEG_EXE}")
            sys.exit(1)
        
        if not os.path.exists(VAD_MODEL_PATH):
            log("Error", f"错误：未找到 VAD 模型 - {VAD_MODEL_PATH}")
            sys.exit(1)
        
        log("Init", "步骤2: 初始化 ASR 引擎...")
        # 初始化 ASR 引擎
        log("Init", "初始化 ASR 引擎...")
        init_asr_engines()
        log("Init", "ASR 引擎初始化完成")
        
        log("Init", "步骤3: 启动后台工作线程...")
        # 启动后台工作线程
        log("Init", "启动后台工作线程...")
        socketio.start_background_task(worker_loop)
        log("Init", "后台工作线程启动完成")
        
        # 检查端口可用性
        port = find_available_port(5231)
        if port != 5231:
            log("Init", f"端口 5231 已被占用，自动切换到端口 {port}")
        
        log("Init", "步骤4: 启动 Web 服务...")
        log("Init", "启动 Web 服务...")
        log("Init", f"服务已启动，请访问 http://127.0.0.1:{port}")
        log("Init", "按 Ctrl+C 停止服务")
        
        # 延迟自动打开浏览器
        import webbrowser
        def _open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")
        threading.Thread(target=_open_browser, daemon=True).start()
        
        log("Init", "步骤5: 开始运行 SocketIO 服务器...")
        socketio.run(app, host="0.0.0.0", port=port, debug=False)
        
    except KeyboardInterrupt:
        log("Info", "用户中断，正在关闭服务...")
    except Exception as e:
        log("Error", f"服务启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 优雅关闭 ASR 连接
        try:
            stop_qwen_asr()
        except Exception:
            pass
        log("Info", "服务已关闭")