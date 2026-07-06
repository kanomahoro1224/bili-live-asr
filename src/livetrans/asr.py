import base64
import logging
import os
import queue
import re
import tempfile
import threading
import wave
from contextlib import nullcontext
from http import HTTPStatus
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("LiveTranslate.ASR")

SAMPLE_RATE = 16000
KOTOBA_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"
DEFAULT_PAD_SECONDS = 0.5
PAD_SECONDS_ENV = "LIVETRANS_SENSEVOICE_PAD_SECONDS"

# Language tag mapping from SenseVoice output
LANG_MAP = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|yue|>": "yue",
}

LANGUAGE_MAP = {
    "auto": "japanese",
    "ja": "japanese",
    "jp": "japanese",
    "japanese": "japanese",
    "zh": "chinese",
    "cn": "chinese",
    "chinese": "chinese",
    "en": "english",
    "english": "english",
    "ko": "korean",
    "kr": "korean",
    "korean": "korean",
}


class KotobaWhisperEngine:
    """Speech-to-text using the same Kotoba Whisper pipeline as KITS-main."""

    def __init__(
        self,
        model_id: str = KOTOBA_MODEL_ID,
        device: str = "cuda",
        language: str = "ja",
        beams: int = 3,
        chunk_length_s: int = 15,
        batch_size: int = 8,
        stride_length_s: tuple[int, int] = (5, 3),
        model_dir: str | None = None,
    ):
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        self.model_id = model_id
        self.model_dir = self._resolve_model_dir(model_id, model_dir)
        self.device = self._normalize_device(device)
        self.language = self._normalize_language(language)
        self.beams = beams

        try:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.model_dir),
                local_dir_use_symlinks=False,
                local_files_only=False,
            )
        except Exception as e:
            log.warning(f"Kotoba model download warning, trying local cache: {e}")

        use_gpu = self.device.startswith("cuda") or self.device == "mps"
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=str(self.model_dir),
            dtype=torch.float16 if use_gpu else torch.float32,
            device=self.device,
            chunk_length_s=chunk_length_s,
            model_kwargs={"attn_implementation": "sdpa"} if use_gpu else {},
            batch_size=batch_size,
            stride_length_s=stride_length_s,
        )
        log.info(
            f"Kotoba Whisper loaded: {self.model_dir} on {self.device} "
            f"(language={self.language}, beams={self.beams})"
        )

    @staticmethod
    def _resolve_model_dir(model_id: str, model_dir: str | None) -> Path:
        if model_dir:
            path = Path(model_dir)
        else:
            path = Path("models") / model_id.rstrip("/").split("/")[-1]
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @staticmethod
    def _normalize_device(device: str) -> str:
        device = str(device or "cuda").lower()
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("未检测到 CUDA，Kotoba Whisper 当前配置要求 asr_device=cuda")
            if ":" in device:
                try:
                    index = int(device.split(":", 1)[1])
                except ValueError as e:
                    raise RuntimeError(f"无效 CUDA 设备: {device}") from e
                if index < 0 or index >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA 设备不存在: {device}")
            return device
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("未检测到 Apple Silicon MPS")
            return "mps"
        return "cpu"

    @staticmethod
    def _normalize_language(language: str) -> str:
        return LANGUAGE_MAP.get(str(language or "ja").lower(), str(language))

    def set_language(self, language: str):
        old = self.language
        self.language = self._normalize_language(language)
        log.info(f"Kotoba Whisper language: {old} -> {self.language}")

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if audio is None or audio.size == 0:
            return None
        result = self._pipe(
            {"array": audio.astype(np.float32, copy=False), "sampling_rate": SAMPLE_RATE},
            return_timestamps=True,
            generate_kwargs={
                "language": self.language,
                "task": "transcribe",
                "num_beams": self.beams,
                "no_repeat_ngram_size": 3,
            },
        )
        text = str(result.get("text") or "").strip()
        if not text and result.get("chunks"):
            text = "".join(str(c.get("text", "")) for c in result["chunks"]).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None
        return {"text": text, "language": self.language, "language_name": self.language}


class SenseVoiceEngine:
    """Speech-to-text using FunASR SenseVoice."""

    def __init__(self, model_name=None, device="cuda", hub="ms", pad_seconds=None):
        from funasr import AutoModel
        from .model_manager import (
            get_local_model_path,
            asr_model_id,
            neutralize_funasr_requirements,
        )

        local = get_local_model_path("sensevoice", hub=hub)
        model = local or model_name or asr_model_id("sensevoice", hub)
        neutralize_funasr_requirements(local)
        self._set_precision(device)
        model_kwargs = {
            "model": model,
            "trust_remote_code": True,
            "device": device,
            "hub": hub,
            "disable_update": True,
        }
        if self._use_fp16:
            model_kwargs["fp16"] = True
        self._model = AutoModel(**model_kwargs)
        device = self._model.kwargs.get("device", device)
        self._set_precision(device)
        self._update_runtime_kwargs(device)
        self._set_input_padding(pad_seconds, log_change=False)
        self.language = None  # None = auto detect
        log.info(
            f"SenseVoice loaded: {model} on {device} "
            f"(hub={hub}, precision={self._precision})"
        )
        self._log_input_padding()

    @staticmethod
    def _read_pad_seconds(value=None) -> float:
        if value is None:
            value = os.environ.get(PAD_SECONDS_ENV)
        if value is None:
            return DEFAULT_PAD_SECONDS
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning(
                f"Invalid SenseVoice pad seconds={value!r}; "
                f"using default {DEFAULT_PAD_SECONDS:g}s"
            )
            return DEFAULT_PAD_SECONDS

    def _set_input_padding(self, pad_seconds=None, log_change=True):
        self._pad_seconds = self._read_pad_seconds(pad_seconds)
        self._pad_quantum = int(round(SAMPLE_RATE * self._pad_seconds))
        if log_change:
            self._log_input_padding()

    def _log_input_padding(self):
        if self._pad_quantum > 0:
            log.info(
                "SenseVoice input padding enabled: "
                f"bucket={self._pad_seconds:g}s, quantum={self._pad_quantum} samples"
            )
        else:
            log.info("SenseVoice input padding disabled")

    @staticmethod
    def _is_cuda_device(device: str) -> bool:
        return str(device).lower().startswith("cuda") and torch.cuda.is_available()

    def _set_precision(self, device: str):
        self._use_fp16 = self._is_cuda_device(device)
        self._precision = "fp16" if self._use_fp16 else "fp32"

    def _apply_model_precision(self):
        model = self._model.model
        if self._use_fp16:
            model.half()
        else:
            model.float()

    def _update_runtime_kwargs(self, device: str):
        self._model.kwargs["device"] = device
        self._model.kwargs["fp16"] = self._use_fp16
        if not self._use_fp16:
            self._model.kwargs.pop("bf16", None)

    def _autocast_context(self):
        if self._use_fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _prepare_audio_input(self, audio: np.ndarray) -> np.ndarray:
        if self._pad_quantum <= 0 or audio.size == 0:
            return audio

        original_samples = audio.shape[0]
        remainder = original_samples % self._pad_quantum
        if remainder == 0:
            return audio

        padded_samples = original_samples + self._pad_quantum - remainder
        padded = np.pad(audio, (0, padded_samples - original_samples), mode="constant")
        log.debug(
            f"SenseVoice input padded: {original_samples} -> {padded_samples} samples"
        )
        return padded

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"SenseVoice language: {old} -> {self.language}")

    def set_input_padding(self, pad_seconds):
        old_quantum = self._pad_quantum
        self._set_input_padding(pad_seconds, log_change=False)
        if self._pad_quantum != old_quantum:
            self._log_input_padding()

    def to_device(self, device: str):
        self._set_precision(device)
        if self._use_fp16:
            self._apply_model_precision()
            self._model.model.to(device)
        else:
            self._model.model.to(device)
            self._apply_model_precision()
        self._update_runtime_kwargs(device)
        log.info(f"SenseVoice moved to {device} (precision={self._precision})")

    def unload(self):
        if hasattr(self, "_model") and self._model is not None:
            try:
                self._model.model.to("cpu")
            except Exception:
                pass
            self._model = None

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono

        Returns:
            dict with 'text', 'language', 'language_name' or None.
        """
        cache = {}
        try:
            audio_input = self._prepare_audio_input(audio)
            with torch.inference_mode(), self._autocast_context():
                result = self._model.generate(
                    input=audio_input,
                    cache=cache,
                    language=self.language or "auto",
                    use_itn=True,
                    batch_size_s=0,
                    disable_pbar=True,
                )
        finally:
            cache.clear()

        if not result or not result[0].get("text"):
            return None

        raw_text = result[0]["text"]

        # Parse language tag and clean text
        detected_lang = "auto"
        text = raw_text

        for tag, lang in LANG_MAP.items():
            if tag in text:
                detected_lang = lang
                text = text.replace(tag, "")
                break

        # Remove emotion/event tags like <|HAPPY|>, <|BGM|>, <|Speech|> etc.
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()

        if not text:
            return None

        log.debug(f"Raw: {raw_text}")
        return {
            "text": text,
            "language": detected_lang,
            "language_name": detected_lang,
        }


class QwenRealtimeEngine:
    """Remote DashScope Qwen realtime ASR with the local ASR transcribe interface."""

    def __init__(
        self,
        api_key: str,
        language: str = "ja",
        model: str = "qwen3-asr-flash-realtime",
        url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        timeout: float = 8.0,
    ):
        if not api_key:
            raise RuntimeError("Qwen ASR requires dashscope_api_key")
        from dashscope.audio.qwen_omni import (
            MultiModality,
            OmniRealtimeCallback,
            OmniRealtimeConversation,
        )
        from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
        import dashscope

        self.api_key = api_key
        self.language = language
        self.model = model
        self.url = url
        self.timeout = float(timeout)
        self._MultiModality = MultiModality
        self._TranscriptionParams = TranscriptionParams
        self._lock = threading.Lock()
        self._results: queue.Queue[str] = queue.Queue()
        dashscope.api_key = api_key

        engine = self

        class _Callback(OmniRealtimeCallback):
            def on_open(self):
                log.info("Qwen ASR connected")

            def on_close(self, code, msg):
                log.info(f"Qwen ASR closed: code={code}, msg={msg}")

            def on_event(self, response):
                event_type = response.get("type")
                if event_type == "conversation.item.input_audio_transcription.completed":
                    text = str(response.get("transcript") or "").strip()
                    if text:
                        engine._results.put(text)

        self._conversation = OmniRealtimeConversation(
            model=self.model,
            url=self.url,
            callback=_Callback(),
        )
        self._conversation.connect()
        self._update_session()
        log.info(f"Qwen ASR loaded: model={self.model}, language={self.language}")

    def _update_session(self):
        params = self._TranscriptionParams(
            language=self.language if self.language != "auto" else "ja",
            sample_rate=SAMPLE_RATE,
            input_audio_format="pcm",
        )
        self._conversation.update_session(
            output_modalities=[self._MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=params,
        )

    def set_language(self, language: str):
        self.language = language or "ja"
        try:
            self._update_session()
        except Exception as e:
            log.warning(f"Qwen ASR language update failed: {e}")

    def unload(self):
        try:
            self._conversation.close()
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if audio is None or audio.size == 0:
            return None
        with self._lock:
            while not self._results.empty():
                try:
                    self._results.get_nowait()
                except queue.Empty:
                    break

            tail_silence = np.zeros(int(SAMPLE_RATE * 0.35), dtype=np.int16).tobytes()
            self._conversation.append_audio(self._encode_audio(audio, tail_silence))

            try:
                text = self._results.get(timeout=self.timeout)
            except queue.Empty:
                return None
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return None
            return {"text": text, "language": self.language, "language_name": self.language}

    def transcribe_stream_frame(self, audio: np.ndarray) -> list[dict]:
        if audio is None or audio.size == 0:
            return []
        with self._lock:
            self._conversation.append_audio(self._encode_audio(audio))
            return [
                {"text": text, "language": self.language, "language_name": self.language}
                for text in self._drain_results()
            ]

    def _drain_results(self) -> list[str]:
        results: list[str] = []
        while True:
            try:
                text = self._results.get_nowait()
            except queue.Empty:
                break
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                results.append(text)
        return results

    def _encode_audio(self, audio: np.ndarray, suffix: bytes = b"") -> str:
        pcm = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
        return base64.b64encode(pcm16 + suffix).decode("ascii")


class DashScopeRemoteEngine:
    """Remote DashScope ASR for one completed VAD segment at a time."""

    def __init__(
        self,
        api_key: str,
        language: str = "ja",
        model: str = "paraformer-realtime-v2",
        timeout: float = 8.0,
    ):
        if not api_key:
            raise RuntimeError("Remote ASR requires dashscope_api_key")
        from dashscope.audio.asr.recognition import Recognition
        import dashscope

        dashscope.api_key = api_key
        self.api_key = api_key
        self.language = language or "auto"
        self.model = model
        self.timeout = float(timeout)
        self._Recognition = Recognition
        log.info(f"DashScope remote ASR loaded: model={self.model}")

    def set_language(self, language: str):
        self.language = language or "auto"

    def transcribe(self, audio: np.ndarray) -> dict | None:
        if audio is None or audio.size == 0:
            return None

        path = self._write_temp_wav(audio)
        try:
            recognizer = self._Recognition(
                model=self.model,
                callback=None,
                format="wav",
                sample_rate=SAMPLE_RATE,
            )
            result = recognizer.call(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

        if getattr(result, "status_code", None) != HTTPStatus.OK:
            code = getattr(result, "code", "")
            message = getattr(result, "message", "")
            raise RuntimeError(f"Remote ASR failed: {code} {message}".strip())

        sentences = result.get_sentence()
        if isinstance(sentences, dict):
            texts = [sentences.get("text", "")]
        elif isinstance(sentences, list):
            texts = [str(s.get("text", "")) for s in sentences if isinstance(s, dict)]
        else:
            texts = []

        text = re.sub(r"\s+", " ", " ".join(t for t in texts if t).strip())
        if not text:
            return None
        return {"text": text, "language": self.language, "language_name": self.language}

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        pcm = np.clip(audio.astype(np.float32, copy=False), -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(SAMPLE_RATE)
            f.writeframes(pcm16.tobytes())
        return path
