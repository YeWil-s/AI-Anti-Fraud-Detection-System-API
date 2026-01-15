"""
音频流处理器
"""
import base64
import io
import numpy as np
from typing import Dict, Optional
import asyncio
# [新增] 导入日志工厂
from app.core.logger import get_logger

# [新增] 初始化模块级 logger
logger = get_logger(__name__)


class AudioProcessor:
    """音频流处理和切片"""
    
    def __init__(self, chunk_duration: int = 3):
        """
        初始化音频处理器
        
        Args:
            chunk_duration: 音频切片时长(秒),默认3秒
        """
        self.chunk_duration = chunk_duration
        self.buffer = {}  # 存储每个用户的音频缓冲
    
    async def process_chunk(self, audio_data: str, user_id: Optional[int] = None) -> Dict:
        """
        处理音频数据块
        """
        try:
            # 解码base64数据
            audio_bytes = base64.b64decode(audio_data)
            
            # TODO: 在这里添加实际的音频处理逻辑
            # 1. 音频格式转换
            # 2. 特征提取(MFCC)
            # 3. 调用AI模型检测
            
            # 当前返回模拟结果
            return {
                "status": "success",
                "chunk_size": len(audio_bytes),
                "timestamp": asyncio.get_event_loop().time(),
                "confidence": 0.95,  # 模拟置信度
                "is_fake": False  # 模拟检测结果
            }
            
        except Exception as e:
            # [新增] 即使没有 print，也必须记录异常堆栈，否则无法排查为什么处理失败
            logger.error(f"Process audio chunk failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def extract_features(self, audio_bytes: bytes) -> np.ndarray:
        """
        提取音频特征(MFCC等)
        """
        # TODO: 使用librosa提取MFCC特征
        # import librosa
        # y, sr = librosa.load(io.BytesIO(audio_bytes))
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 当前返回模拟数据
        return np.random.rand(13, 100)
    
    def add_to_buffer(self, user_id: int, audio_chunk: bytes):
        """添加音频块到缓冲区"""
        if user_id not in self.buffer:
            self.buffer[user_id] = []
        self.buffer[user_id].append(audio_chunk)
    
    def get_buffered_audio(self, user_id: int) -> Optional[bytes]:
        """获取用户的缓冲音频并清空缓冲区"""
        if user_id in self.buffer:
            audio = b''.join(self.buffer[user_id])
            self.buffer[user_id] = []
            return audio
        return None
    
    def clear_buffer(self, user_id: int):
        """清空指定用户的缓冲区"""
        if user_id in self.buffer:
            del self.buffer[user_id]