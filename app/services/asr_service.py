"""
本地语音转文本服务 (ASR)
使用 faster-whisper (Tiny 模型，支持纯 CPU 高速推理)
"""
import os
import tempfile
from faster_whisper import WhisperModel
from app.core.logger import get_logger

logger = get_logger(__name__)

class ASRService:
    def __init__(self):
        logger.info(" 正在初始化本地 ASR 语音识别引擎 (Whisper-Tiny)...")
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    async def transcribe_audio_file(self, file_path: str) -> str:
        """
        读取本地音频/视频文件，转换为文本
        """
        try:
            # beam_size=5 保证翻译的准确率，language="zh" 强制中文识别
            segments, info = self.model.transcribe(file_path, beam_size=5, language="zh")
            text = "".join([segment.text for segment in segments])
            logger.info(f"🎤 ASR 提取文本成功: {text}")
            return text
        except Exception as e:
            logger.error(f"❌ ASR 提取失败: {e}")
            return ""

asr_service = ASRService()