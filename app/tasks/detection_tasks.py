"""
AI检测异步任务 (Celery Worker)
适配 main.py 的 Redis 监听模式
集成第一阶段Day 5：结果防抖与状态机 (Redis版)
"""
import redis  
import json   
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
# =========================================================
def publish_to_redis(user_id: int, payload: dict):
    try:
        r = redis.from_url(settings.REDIS_URL)
        message_data = {
            "user_id": user_id,
            "payload": payload
        }
        r.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish to Redis: {e}")

# =========================================================
# [新增] 视频结果防抖与状态机逻辑
# =========================================================
def apply_video_debounce(user_id: int, call_id: int, raw_is_fake: bool) -> dict:
    """
    使用 Redis 实现滑动窗口防抖和状态机
    策略：
    1. 窗口大小 5帧
    2. 报警阈值 >= 3帧为假 (进入 ALARM)
    3. 解除阈值 <= 1帧为假 (回到 SAFE)
    4. 中间状态保持不变 (Hysteresis 滞后效应，防止跳变)
    """
    try:
        r = redis.from_url(settings.REDIS_URL)
        
        # 定义 Key
        window_key = f"detect:window:{call_id}"  # 存最近5次结果 [1, 0, 1...]
        state_key = f"detect:state:{call_id}"    # 存当前状态 "SAFE" 或 "ALARM"
        
        # 1. 写入本次结果 (1=Fake, 0=Real)
        val = 1 if raw_is_fake else 0
        r.lpush(window_key, val)
        r.ltrim(window_key, 0, 4) # 只保留最近5个
        r.expire(window_key, 3600) # 设置过期时间防止脏数据
        
        # 2. 读取窗口并统计
        # lrange 返回的是字符串列表 ['1', '0'...]
        history = r.lrange(window_key, 0, -1)
        fake_count = sum(int(x) for x in history)
        
        # 3. 获取当前状态
        current_state = r.get(state_key)
        if current_state:
            current_state = current_state.decode('utf-8')
        else:
            current_state = "SAFE"
            
        # 4. 状态机逻辑 (核心)
        final_state = current_state 
        
        if fake_count >= 3:
            final_state = "ALARM"
        elif fake_count <= 1:
            final_state = "SAFE"
        # else: fake_count == 2 时，保持 current_state 不变 (防抖关键)
        
        # 更新状态
        if final_state != current_state:
            r.setex(state_key, 3600, final_state)
            
        return {
            "final_is_fake": (final_state == "ALARM"), # 最终对外输出
            "fake_count": fake_count,
            "state": final_state
        }
        
    except Exception as e:
        logger.error(f"Debounce logic failed: {e}")
        # 如果 Redis 挂了，降级为直接信任本次结果
        return {"final_is_fake": raw_is_fake, "state": "UNKNOWN", "fake_count": -1}


@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_base64: str, user_id: int, call_id: int) -> Dict:
    """音频检测任务 (代码保持不变)"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect audio (Len: {len(audio_base64)})")

    try:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return {"status": "error", "message": "Invalid base64"}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            if settings.COLLECT_TRAINING_DATA:
                loop.run_until_complete(save_raw_data(audio_bytes, user_id, call_id, "audio", "wav"))

            self.update_state(state='PROCESSING', meta={'progress': 50})
            result = loop.run_until_complete(model_service.predict_voice(audio_bytes))
            
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
    """视频检测任务 (已修改：增加防抖)"""
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

            # 3. 推理 (获取原始结果)
            self.update_state(state='PROCESSING', meta={'progress': 50})
            raw_result = loop.run_until_complete(model_service.predict_video(video_tensor))
            raw_is_fake = raw_result.get('is_deepfake', False)
            raw_conf = raw_result.get('confidence', 0.0)

            # 4. [修改点] 应用防抖与状态机逻辑
            # 不再直接使用 raw_is_fake，而是使用 debounce_data['final_is_fake']
            debounce_data = apply_video_debounce(user_id, call_id, raw_is_fake)
            final_is_fake = debounce_data['final_is_fake']
            
            logger.info(f"Video Check -> Raw: {raw_is_fake}, Final: {final_is_fake} "
                        f"(Win: {debounce_data.get('fake_count')}/5, State: {debounce_data.get('state')})")

            # 5. 根据“防抖后的结果”发布消息
            if final_is_fake:
                logger.warning(f"⚠️ DEEPFAKE CONFIRMED (Stable)! Conf: {raw_conf}")
                
                payload = {
                    "type": "alert",
                    "msg_type": "video",
                    "message": "检测到Deepfake视频流!",
                    # 依然发送原始置信度供参考
                    "confidence": raw_conf,
                    "risk_level": "high",
                    "call_id": call_id,
                    "debug_info": f"window_fake:{debounce_data.get('fake_count')}/5"
                }
                publish_to_redis(user_id, payload)
            else:
                logger.info("Video Safe (Stable)")
                payload = {
                    "type": "info",
                    "msg_type": "video",
                    "message": "画面正常",
                    "confidence": raw_conf,
                    "call_id": call_id,
                    "debug_info": f"window_fake:{debounce_data.get('fake_count')}/5"
                }
                publish_to_redis(user_id, payload)

            return {"status": "success", "result": raw_result, "debounce": debounce_data}
        finally:
            loop.close()
    except Exception as e:
        logger.error(f"Video task failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """文本检测任务 (代码保持不变)"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect text (Len: {len(text)})")

    try:
        self.update_state(state='PROCESSING', meta={'progress': 0})
        result = model_service.predict_text(text)
        self.update_state(state='PROCESSING', meta={'progress': 80})
        
        if result.get('label') == 'fraud':
            logger.warning(f"⚠️ DETECTED SCAM TEXT! Conf: {result.get('confidence')}")
            payload = {
                "type": "alert",
                "msg_type": "text",
                "message": "检测到诈骗话术!",
                "confidence": result['confidence'],
                "keywords": result.get('keywords', []),
                "call_id": call_id
            }
            publish_to_redis(user_id, payload)
        else:
            logger.info(f"Text detection passed. Conf: {result.get('confidence')}")
            payload = {
                "type": "info",
                "msg_type": "text",
                "message": "语义安全",
                "confidence": result.get('confidence'),
                "call_id": call_id
            }
            publish_to_redis(user_id, payload)

        return {"status": "success", "result": result, "call_id": call_id}
    except Exception as e:
        logger.error(f"Text detection task failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "call_id": call_id}

@celery_app.task(name="get_task_status")
def get_task_status(task_id: str) -> Dict:
    res = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": res.status, "result": res.result if res.successful() else None}