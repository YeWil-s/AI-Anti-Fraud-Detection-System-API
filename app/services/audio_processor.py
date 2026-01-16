"""
音频流处理器
"""
import base64
import io
import numpy as np
import librosa  # [新增] 引入音频处理库
from typing import Dict, Optional, List
import asyncio
from app.core.logger import get_logger

# 初始化模块级 logger
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
            
            # [修改] 尝试提取特征以验证音频有效性
            # 注意：实际的检测逻辑通常在 Celery 任务中通过 model_service 完成
            # 这里主要是确保上传的音频格式正确且不为空
            features = await self.extract_features(audio_bytes)
            
            if features is None:
                return {
                    "status": "warning",
                    "message": "Audio extract failed or silent"
                }
            
            # 返回基本信息，特征提取在检测任务中会再次调用(或者你可以选择在这里提取完传给Celery)
            return {
                "status": "success",
                "chunk_size": len(audio_bytes),
                "timestamp": asyncio.get_event_loop().time(),
                "confidence": 0.0,  # 占位符，实际结果由 Celery 异步推送
                "is_fake": False
            }
            
        except Exception as e:
            # 记录异常堆栈
            logger.error(f"Process audio chunk failed: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def extract_features(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """
        提取音频特征 (MFCC) - 对接 GNB/NMF 统计流模型
        """
        try:
            # 1. 加载音频
            # io.BytesIO 将内存中的 bytes 转为文件对象供 librosa 读取
            # sr=16000: 统一重采样到 16k，这是语音处理的标准采样率
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            
            # 过滤过短或静音音频
            if len(y) < 100: 
                return None

            # 2. 提取 MFCC
            # n_mfcc=20: 论文中常用的参数，用于捕捉精细的频谱包络
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # 3. 转置特征矩阵 (关键步骤!)
            # librosa 输出形状是 (n_mfcc, n_frames) -> (20, Time)
            # Sklearn (GNB/SVM) 模型需要的输入是 (n_samples, n_features) -> (Time, 20)
            return mfcc.T
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
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