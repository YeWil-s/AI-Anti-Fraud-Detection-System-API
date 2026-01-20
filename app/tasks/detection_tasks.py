"""
AI检测异步任务
"""
from app.tasks.celery_app import celery_app
from app.services.model_service import model_service
from app.services.websocket_manager import connection_manager
# [新增] 引入音频处理器，用于在 Celery 内部提取特征
from app.services.audio_processor import AudioProcessor
from app.core.storage import upload_to_minio
from app.core.config import settings
import time
import io

import numpy as np
import base64
from typing import Dict, List, Union
import asyncio
from app.core.logger import get_logger, bind_context

# 初始化模块级 logger
logger = get_logger(__name__)

# [新增] 实例化一个全局音频处理器工具 (专门用于 Celery 内部提取特征)
celery_audio_processor = AudioProcessor()

async def save_raw_data(data: bytes, user_id: int, call_id: int, data_type: str, ext: str):
    """
    将原始数据保存到 MinIO 的 dataset 目录
    路径格式: dataset/{type}/{user_id}/{call_id}_{timestamp}.{ext}
    """
    if not settings.COLLECT_TRAINING_DATA:
        return

    try:
        timestamp = int(time.time() * 1000) # 毫秒级时间戳
        filename = f"dataset/{data_type}/{user_id}/{call_id}_{timestamp}.{ext}"
        
        # 这里的 content_type 可以稍微宽泛一点
        content_type = "audio/wav" if data_type == "audio" else "application/octet-stream"
        
        # 上传 (不等待返回 URL，不需要阻塞太久)
        await upload_to_minio(data, filename, content_type=content_type)
        logger.debug(f"💾 Raw data collected: {filename}")
        
    except Exception as e:
        # 数据采集失败不应影响主业务，仅记录 Warning
        logger.warning(f"Failed to collect training data: {e}")

@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_base64: str, user_id: int, call_id: int) -> Dict:
    """
    音频检测异步任务 (Scheme B: 接收原始音频 -> Celery内部提取MFCC -> 推理)
    
    Args:
        audio_base64: Base64编码的原始音频数据
    """
    bind_context(user_id=user_id, call_id=call_id)
    # 日志仅记录数据长度，避免打印 Base64
    logger.info(f"Task started: Detect audio (Data len: {len(audio_base64)})")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 10})
        
        # 1. Base64 解码
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            return {"status": "error", "message": "Invalid base64 data"}

        # 2. 特征提取 (耗时操作，放在 Celery 中执行)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 2. 数据采集 (保存原始音频)
            # 放在特征提取之前，确保即使提取失败也能拿到原始数据用于分析原因
            if settings.COLLECT_TRAINING_DATA:
                loop.run_until_complete(
                    save_raw_data(audio_bytes, user_id, call_id, "audio", "wav")
                )

            #   特征提取
            mfcc_features = loop.run_until_complete(
                celery_audio_processor.extract_features(audio_bytes)
            )
            
            if mfcc_features is None:
                logger.warning("Audio feature extraction failed or audio too short")
                return {"status": "skipped", "reason": "Feature extraction failed"}
            
            self.update_state(state='PROCESSING', meta={'progress': 50})

            # 3. 执行 AI 检测 (传入 MFCC 特征)
            result = loop.run_until_complete(model_service.predict_voice(mfcc_features))
            
            self.update_state(state='PROCESSING', meta={'progress': 90})
            
            # 4. 结果处理与推送
            if result.get('is_fake'):
                logger.warning(f"⚠️ DETECTED FAKE AUDIO! Confidence: {result.get('confidence')}")
                
                loop.run_until_complete(
                    connection_manager.send_personal_message({
                        "type": "alert",
                        "msg_type": "audio",
                        "message": "检测到伪造语音!",
                        "confidence": result['confidence'],
                        "details": result.get('details'),
                        "call_id": call_id
                    }, user_id)
                )
            else:
                logger.info(f"Audio detection passed (Real). Conf: {result.get('confidence')}")

            return {
                "status": "success",
                "result": result,
                "call_id": call_id
            }

        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Audio detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }


@celery_app.task(name="detect_video", bind=True)
def detect_video_task(self, frame_data: list, user_id: int, call_id: int) -> Dict:
    """
    视频检测异步任务 (Scheme A: 接收时序 Tensor List -> 推理)
    
    Args:
        frame_data: List 格式的 5D Tensor (1, 10, 3, 224, 224)
    """
    bind_context(user_id=user_id, call_id=call_id)
    logger.info("Task started: Detect video batch (Sequence)")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        # 1. 还原为 Numpy Tensor
        # [关键] 必须指定 float32，否则 ONNX Runtime 可能会报错 (Type Error)
        video_tensor = np.array(frame_data, dtype=np.float32)
        
        # 简单校验形状
        if len(video_tensor.shape) != 5:
            logger.error(f"Invalid video tensor shape: {video_tensor.shape}, expected 5D")
            return {"status": "error", "message": "Invalid tensor shape"}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # [新增] 2. 数据采集 (保存处理后的 Tensor 数据)
            # 因为目前传入的是已经预处理过的 Tensor，我们先存为 .npy 格式
            # 虽然不是最原始的图片，但对于调试 "Tensor -> Model" 这个环节非常有帮助
            if settings.COLLECT_TRAINING_DATA:
                # 使用 BytesIO 在内存中保存 npy
                buffer = io.BytesIO()
                np.save(buffer, video_tensor)
                npy_bytes = buffer.getvalue()
                
                loop.run_until_complete(
                    save_raw_data(npy_bytes, user_id, call_id, "video_tensor", "npy")
                )

            # 3. 执行 AI 检测
            result = loop.run_until_complete(model_service.predict_video(video_tensor))
            
            self.update_state(state='PROCESSING', meta={'progress': 80})
            
            # 3. 结果推送
            if result.get('is_deepfake'):
                logger.warning(f"⚠️ DETECTED DEEPFAKE VIDEO! Confidence: {result.get('confidence')}")
                
                loop.run_until_complete(
                    connection_manager.send_personal_message({
                        "type": "alert",
                        "msg_type": "video",
                        "message": "检测到Deepfake视频流!",
                        "confidence": result['confidence'],
                        "call_id": call_id
                    }, user_id)
                )
            else:
                logger.info("Video detection passed (Real)")

            return {
                "status": "success",
                "result": result,
                "call_id": call_id
            }

        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Video detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }


@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """
    文本检测异步任务 (BERT)
    """
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect text (Len: {len(text)})")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(model_service.predict_text(text))
            
            self.update_state(state='PROCESSING', meta={'progress': 80})
            
            if result.get('is_scam'):
                logger.warning(f"⚠️ DETECTED SCAM TEXT! Conf: {result.get('confidence')}")
                
                loop.run_until_complete(
                    connection_manager.send_personal_message({
                        "type": "alert",
                        "msg_type": "text",
                        "message": "检测到诈骗话术!",
                        "confidence": result['confidence'],
                        "keywords": result.get('keywords', []),
                        "call_id": call_id
                    }, user_id)
                )
            else:
                logger.info("Text detection passed (Normal)")

            return {
                "status": "success",
                "result": result,
                "call_id": call_id
            }

        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Text detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }

# get_task_status 保持不变...
@celery_app.task(name="get_task_status")
def get_task_status(task_id: str) -> Dict:
    # ... (原有代码) ...
    task_result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.successful() else None,
        "info": task_result.info
    }