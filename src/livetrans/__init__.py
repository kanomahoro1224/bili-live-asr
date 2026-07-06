"""livetrans —— 本地 ASR 实时直播翻译（分层包）。

分层依赖方向（纯逻辑在下、重依赖在上、编排在顶）：
config / logging_util / filters  →  stream / audio / translator
  →  vad / asr（重依赖 torch/funasr，延迟导入）  →  pipeline  →  web / server
"""

__all__ = ["__version__"]

__version__ = "2.0.0-local"
