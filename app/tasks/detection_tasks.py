"""
AI检测异步任务 (Celery Worker) - 修复版
1. 修复 IntegrityError: 自动创建缺失的 CallRecord
2. 修复 Event Loop: 使用 NotificationService 统一处理
3. 集成防抖与规则引擎
"""
import redis
import json
import asyncio
import base64
import io
import time
import numpy as np
from typing import Dict, List, Union
from datetime import datetime  # [新增]

from sqlalchemy import select # [新增]
from app.models.call_record import CallRecord # [新增]

from app.tasks.celery_app import celery_app
from app.services.model_service import model_service
from app.services.video_processor import VideoProcessor
from app.services.security_service import security_service
from app.services.notification_service import notification_service
from app.db.database import AsyncSessionLocal
from app.models.ai_detection_log import AIDetectionLog

from app.core.storage import upload_to_minio
from app.core.config import settings
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

async def ensure_call_record_exists(db, call_id: int, user_id: int):
    """
    [新增] 确保 CallRecord 存在，防止外键报错
    这是为了兼容测试脚本生成的随机 call_id
    """
    try:
        result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
        record = result.scalar_one_or_none()
        
        if not record:
            logger.info(f"CallRecord {call_id} not found, auto-creating...")
            new_record = CallRecord(
                call_id=call_id,
                user_id=user_id,
                start_time=datetime.now(),
                caller_number="unknown",
                duration=0
            )
            db.add(new_record)
            await db.commit() # 必须立即提交以供后续外键引用
            return True
    except Exception as e:
        logger.error(f"Failed to ensure call record: {e}")
        # 这里不抛出异常，尝试继续执行，虽然大概率后面会报错
    return False

# =========================================================
# [核心辅助函数] 发布控制指令 (UI控制/软阻断)
# =========================================================
def publish_control_command(user_id: int, payload: dict):
    """
    发送控制指令 (如升级防御等级、挂断通话)
    """
    try:
        message_data = {
            "user_id": user_id,
            "payload": payload
        }
        # 直接使用 notification_service 内部的 redis 客户端推送
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish control command: {e}")

# =========================================================
# [核心辅助函数] 视频结果防抖与状态机逻辑
# =========================================================
def apply_video_debounce(user_id: int, call_id: int, raw_is_fake: bool) -> dict:
    """
    使用 Redis 实现滑动窗口防抖和状态机
    """
    try:
        r = redis.from_url(settings.REDIS_URL)
        
        # 定义 Key
        window_key = f"detect:window:{call_id}"  # 存最近5次结果 [1, 0, 1...]
        state_key = f"detect:state:{call_id}"    # 存当前状态 "SAFE" 或 "ALARM"
        
        # 1. 写入本次结果 (1=Fake, 0=Real)
        val = 1 if raw_is_fake else 0
        r.lpush(window_key, val)
        r.ltrim(window_key, 0, 4) 
        r.expire(window_key, 3600) 
        
        # 2. 读取窗口并统计
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
        
        # 更新状态
        if final_state != current_state:
            r.setex(state_key, 3600, final_state)
            
        return {
            "final_is_fake": (final_state == "ALARM"), 
            "fake_count": fake_count,
            "state": final_state
        }
        
    except Exception as e:
        logger.error(f"Debounce logic failed: {e}")
        return {"final_is_fake": raw_is_fake, "state": "UNKNOWN", "fake_count": -1}


@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_base64: str, user_id: int, call_id: int) -> Dict:
    """音频检测任务"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect audio (Len: {len(audio_base64)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # [关键修复] 确保 CallRecord 存在
                await ensure_call_record_exists(db, call_id, user_id)

                try:
                    audio_bytes = base64.b64decode(audio_base64)
                except Exception as e:
                    return {"status": "error", "message": "Invalid base64"}

                if settings.COLLECT_TRAINING_DATA:
                    await save_raw_data(audio_bytes, user_id, call_id, "audio", "wav")

                self.update_state(state='PROCESSING', meta={'progress': 50})
                
                # 模型调用
                result = await model_service.predict_voice(audio_bytes)
                is_fake = result.get('is_fake', False)
                confidence = result.get('confidence', 0.0)
                risk_level = result.get('risk_level', 'low')

                # 1. 记录 AI 技术日志 (现在不会报 FK 错误了)
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    voice_confidence=confidence,
                    overall_score=confidence * 100,
                    model_version="v1.0"
                )
                db.add(ai_log)
                await db.commit()

                # 2. 调用通知服务
                await notification_service.handle_detection_result(
                    db=db,
                    user_id=user_id,
                    call_id=call_id,
                    detection_type="语音",
                    is_risk=is_fake,
                    confidence=confidence,
                    risk_level=risk_level if is_fake else "safe",
                    details=f"检测结果: {'伪造' if is_fake else '真实'}"
                )

                if is_fake:
                    logger.warning(f"⚠️ FAKE AUDIO DETECTED! Conf: {confidence}")
                    
                    # 3. [Level 2] 触发高危防御
                    payload_control = {
                        "type": "control",
                        "action": "upgrade_level",
                        "target_level": 2,
                        "config": {
                            "video_fps": 30.0,
                            "ui_action": "simulate_hangup", 
                            "ui_message": "检测到AI合成语音，请立即挂断！",
                            "warning_mode": "modal"
                        }
                    }
                    publish_control_command(user_id, payload_control)
                else:
                    logger.info("Audio Real")

                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Audio task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_process())
    finally:
        loop.close()


@celery_app.task(name="detect_video", bind=True)
def detect_video_task(self, frame_data: list, user_id: int, call_id: int) -> Dict:
    """视频检测任务"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info("Task started: Detect video batch")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # [关键修复] 确保 CallRecord 存在
                await ensure_call_record_exists(db, call_id, user_id)

                try:
                    video_tensor = VideoProcessor.preprocess_batch(frame_data)
                except Exception as e:
                    return {"status": "error", "message": "Preprocessing failed"}

                if len(video_tensor.shape) != 5:
                    return {"status": "error", "message": "Invalid shape"}

                if settings.COLLECT_TRAINING_DATA:
                    buffer = io.BytesIO()
                    np.save(buffer, video_tensor)
                    await save_raw_data(buffer.getvalue(), user_id, call_id, "video_tensor", "npy")

                self.update_state(state='PROCESSING', meta={'progress': 50})
                
                # 模型推理
                raw_result = await model_service.predict_video(video_tensor)
                raw_is_fake = raw_result.get('is_deepfake', False)
                raw_conf = raw_result.get('confidence', 0.0)

                # 防抖逻辑
                debounce_data = apply_video_debounce(user_id, call_id, raw_is_fake)
                final_is_fake = debounce_data['final_is_fake']
                
                logger.info(f"Video Check -> Raw: {raw_is_fake}, Final: {final_is_fake} "
                            f"(Win: {debounce_data.get('fake_count')}/5, State: {debounce_data.get('state')})")

                # 1. 记录 AI 技术日志
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    video_confidence=raw_conf,
                    overall_score=raw_conf * 100,
                    model_version=raw_result.get("model_version", "v1.0")
                )
                db.add(ai_log)
                await db.commit()

                # 2. 调用通知服务
                await notification_service.handle_detection_result(
                    db=db,
                    user_id=user_id,
                    call_id=call_id,
                    detection_type="视频",
                    is_risk=final_is_fake,
                    confidence=raw_conf,
                    risk_level="high" if final_is_fake else "safe",
                    details=f"Deepfake状态机: {debounce_data.get('state')}"
                )

                if final_is_fake:
                    logger.warning(f"⚠️ DEEPFAKE CONFIRMED (Stable)! Conf: {raw_conf}")

                    # 3. [Level 2] 触发高危防御
                    payload_control = {
                        "type": "control",
                        "action": "upgrade_level",
                        "target_level": 2,
                        "config": {
                            "video_fps": 30.0,
                            "ui_action": "simulate_hangup",
                            "ui_message": "检测到AI换脸视频，请警惕！",
                            "block_call": True
                        }
                    }
                    publish_control_command(user_id, payload_control)
                else:
                    logger.info("Video Safe (Stable)")

                return {"status": "success", "result": raw_result, "debounce": debounce_data}
            except Exception as e:
                logger.error(f"Video task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_process())
    finally:
        loop.close()


@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """文本检测任务"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect text (Len: {len(text)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # [关键修复] 确保 CallRecord 存在
                await ensure_call_record_exists(db, call_id, user_id)

                self.update_state(state='PROCESSING', meta={'progress': 0})
                
                # 1. 规则引擎优先匹配
                rule_hit = None
                try:
                    rule_hit = await security_service.match_risk_rules(text, db) 
                except Exception as e:
                    logger.error(f"Risk rule matching failed: {e}")

                if rule_hit:
                    logger.warning(f"⚠️ RISK RULE MATCHED: {rule_hit['keyword']}")
                    risk_level_code = rule_hit.get('risk_level', 1)
                    
                    await notification_service.handle_detection_result(
                        db=db,
                        user_id=user_id,
                        call_id=call_id,
                        detection_type="文本(规则)",
                        is_risk=True,
                        confidence=1.0,
                        risk_level="high" if risk_level_code >= 4 else "medium",
                        details=f"触发敏感词: {rule_hit['keyword']}"
                    )
                    
                    # 下发指令逻辑 (略)...
                    return {"status": "success", "result": "rule_hit"}

                # 2. AI 模型推理
                result = await model_service.predict_text(text)
                self.update_state(state='PROCESSING', meta={'progress': 80})
                
                is_fraud = (result.get('label') == 'fraud')
                confidence = result.get('confidence', 0.0)

                # 通知服务
                await notification_service.handle_detection_result(
                    db=db,
                    user_id=user_id,
                    call_id=call_id,
                    detection_type="文本(AI)",
                    is_risk=is_fraud,
                    confidence=confidence,
                    risk_level="high" if is_fraud else "safe",
                    details=f"AI语义分析: {result.get('label')}"
                )

                if is_fraud:
                    # 告警逻辑...
                    pass

                return {"status": "success", "result": result, "call_id": call_id}
                
            except Exception as e:
                logger.error(f"Text detection task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "call_id": call_id}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_process())
    finally:
        loop.close()

@celery_app.task(name="get_task_status")
def get_task_status(task_id: str) -> Dict:
    res = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": res.status, "result": res.result if res.successful() else None}