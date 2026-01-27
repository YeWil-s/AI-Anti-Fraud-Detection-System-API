"""
AI检测异步任务 (Celery Worker)
适配 main.py 的 Redis 监听模式
"""
import redis  # [必须] 用于发布 Redis 消息
import json   # [必须] 用于序列化
from app.tasks.celery_app import celery_app
from app.services.model_service import model_service
from app.services.video_processor import VideoProcessor
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

async def save_raw_data(data: bytes, user_id: int, call_id: int, data_type: str, ext: str):
    """保存原始数据到 MinIO"""
    if not settings.COLLECT_TRAINING_DATA:
        return
    try:
        timestamp = int(time.time() * 1000)
        filename = f"dataset/{data_type}/{user_id}/{call_id}_{timestamp}.{ext}"
        content_type = "audio/wav" if data_type == "audio" else "application/octet-stream"
        await upload_to_minio(data, filename, content_type=content_type)
    except Exception as e:
        logger.warning(f"Failed to collect training data: {e}")

# =========================================================
# [核心辅助函数] 发布消息到 Redis
# 这就是 Celery 和 FastAPI 说话的唯一方式
# =========================================================
def publish_to_redis(user_id: int, payload: dict):
    try:
        # 使用同步 Redis 客户端 (Celery Worker 通常是同步环境)
        # 必须确保 settings.REDIS_URL 是正确的 (如 redis://localhost:6379/0)
        r = redis.from_url(settings.REDIS_URL)
        
        # 构造 main.py 中 redis_listener 期待的数据格式
        message_data = {
            "user_id": user_id,
            "payload": payload
        }
        
        # 发布到 'fraud_alerts' 频道 (必须与 main.py 监听的频道一致)
        r.publish("fraud_alerts", json.dumps(message_data))
        # logger.debug(f"Message published to Redis for user {user_id}")
        
    except Exception as e:
        logger.error(f"Failed to publish to Redis: {e}")


@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_base64: str, user_id: int, call_id: int) -> Dict:
    """音频检测任务"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect audio (Len: {len(audio_base64)})")

    try:
        # 1. 解码
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return {"status": "error", "message": "Invalid base64"}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 2. 存数据
            if settings.COLLECT_TRAINING_DATA:
                loop.run_until_complete(save_raw_data(audio_bytes, user_id, call_id, "audio", "wav"))

            # 3. 推理
            self.update_state(state='PROCESSING', meta={'progress': 50})
            result = loop.run_until_complete(model_service.predict_voice(audio_bytes))
            
            # 4. [关键] 发布结果到 Redis
            if result.get('is_fake'):
                logger.warning(f"⚠️ FAKE AUDIO DETECTED! Conf: {result.get('confidence')}")
                
                payload = {
                    "type": "alert",
                    "msg_type": "audio",
                    "message": "检测到伪造语音!",
                    "confidence": result['confidence'],
                    "risk_level": result.get('risk_level', 'high'),
                    "call_id": call_id
                }
                publish_to_redis(user_id, payload)
            else:
                logger.info("Audio Real")
                # 可选: 发送 info 消息
                payload = {
                    "type": "info",
                    "msg_type": "audio",
                    "message": "真实语音",
                    "confidence": result.get('confidence'),
                    "call_id": call_id
                }
                publish_to_redis(user_id, payload)

            return {"status": "success", "result": result}
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Audio task failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="detect_video", bind=True)
def detect_video_task(self, frame_data: list, user_id: int, call_id: int) -> Dict:
    """视频检测任务"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info("Task started: Detect video batch")

    try:
        # 1. 转 Tensor
        try:
            video_tensor = VideoProcessor.preprocess_batch(frame_data)
        except Exception as e:
            return {"status": "error", "message": "Preprocessing failed"}

        if len(video_tensor.shape) != 5:
            return {"status": "error", "message": "Invalid shape"}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 2. 存数据
            if settings.COLLECT_TRAINING_DATA:
                buffer = io.BytesIO()
                np.save(buffer, video_tensor)
                loop.run_until_complete(
                    save_raw_data(buffer.getvalue(), user_id, call_id, "video_tensor", "npy")
                )

            # 3. 推理
            self.update_state(state='PROCESSING', meta={'progress': 50})
            result = loop.run_until_complete(model_service.predict_video(video_tensor))
            
            # 4. [关键] 发布结果到 Redis
            if result.get('is_deepfake'):
                logger.warning(f"⚠️ DEEPFAKE VIDEO DETECTED! Conf: {result.get('confidence')}")
                
                payload = {
                    "type": "alert",
                    "msg_type": "video",
                    "message": "检测到Deepfake视频流!",
                    "confidence": result['confidence'],
                    "risk_level": result.get('risk_level', 'high'),
                    "call_id": call_id
                }
                publish_to_redis(user_id, payload)
            else:
                logger.info("Video Real")
                payload = {
                    "type": "info",
                    "msg_type": "video",
                    "message": "真实画面",
                    "confidence": result.get('confidence'),
                    "call_id": call_id
                }
                publish_to_redis(user_id, payload)

            return {"status": "success", "result": result}
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Video task failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """
    文本检测异步任务 (BERT)
    """
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect text (Len: {len(text)})")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 0})
        
        # [修正 1] predict_text 是同步函数，直接调用，不要用 loop.run_until_complete
        # 这一步是 CPU 密集型的，会阻塞 worker 一小会儿，但在 Celery 里是可以接受的
        result = model_service.predict_text(text)
        
        self.update_state(state='PROCESSING', meta={'progress': 80})
        
        # [修正 2] 检查返回结果的键值
        # model_service.predict_text 返回的是 {"label": "fraud", ...}
        if result.get('label') == 'fraud':
            logger.warning(f"⚠️ DETECTED SCAM TEXT! Conf: {result.get('confidence')}")
            
            # [修正 3] 发送消息是异步的，需要这就需要 Event Loop 了
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
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
            finally:
                loop.close()
        else:
            logger.info(f"Text detection passed (Normal). Conf: {result.get('confidence')}")

        return {
            "status": "success",
            "result": result,
            "call_id": call_id
        }
        
    except Exception as e:
        logger.error(f"Text detection task failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "call_id": call_id
        }

# 状态查询任务保持不变...
@celery_app.task(name="get_task_status")
def get_task_status(task_id: str) -> Dict:
    res = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": res.status, "result": res.result if res.successful() else None}