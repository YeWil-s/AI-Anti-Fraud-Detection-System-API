"""
AI检测异步任务 (Celery Worker) - 深度优化版
1. 消息通知：边缘触发 + 10s冷却 (防止刷屏)
2. 技术日志：动态采样 (Safe=10s/次, Risk=0.5s/次)
3. 状态机防抖：集成 Redis 滑动窗口
"""
import redis
import json
import asyncio
import base64
import io
import time
import numpy as np
from typing import Dict, List, Union, Optional
from datetime import datetime

from sqlalchemy import select
from app.models.call_record import CallRecord

from app.services.llm_service import llm_service
from app.models.user import User
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

async def ensure_call_record_exists(db, call_id: int, user_id: int) -> datetime:
    """
    确保 CallRecord 存在，并返回通话开始时间(start_time)用于计算偏移量
    """
    try:
        result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
        record = result.scalar_one_or_none()
        
        if record:
            # 确保返回 datetime 对象 (如果是 None 则用当前时间)
            return record.start_time or datetime.now()
        
        logger.info(f"CallRecord {call_id} not found, auto-creating...")
        now = datetime.now()
        new_record = CallRecord(
            call_id=call_id,
            user_id=user_id,
            start_time=now,
            caller_number="unknown",
            duration=0
        )
        db.add(new_record)
        await db.commit()
        return now
    except Exception as e:
        logger.error(f"Failed to ensure call record: {e}")
        return datetime.now()

# =========================================================
# [核心辅助函数] 发布控制指令
# =========================================================
def publish_control_command(user_id: int, payload: dict):
    """发送控制指令 (如升级防御等级、挂断通话)"""
    try:
        message_data = {"user_id": user_id, "payload": payload}
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish control command: {e}")

# =========================================================
# [核心辅助函数] 视频结果防抖与状态机逻辑
# =========================================================
def apply_video_debounce(user_id: int, call_id: int, raw_is_fake: bool) -> dict:
    """
    使用 Redis 实现滑动窗口防抖和状态机
    返回: {final_is_fake, fake_count, state, prev_state}
    """
    try:
        r = redis.from_url(settings.REDIS_URL)
        
        # 定义 Key
        window_key = f"detect:window:{call_id}"
        state_key = f"detect:state:{call_id}"
        
        # 1. 写入本次结果 (1=Fake, 0=Real)
        val = 1 if raw_is_fake else 0
        r.lpush(window_key, val)
        r.ltrim(window_key, 0, 4) # 保留最近5帧
        r.expire(window_key, 3600)
        
        # 2. 读取窗口并统计
        history = r.lrange(window_key, 0, -1)
        fake_count = sum(int(x) for x in history)
        
        # 3. 获取上一状态
        prev_state_bytes = r.get(state_key)
        prev_state = prev_state_bytes.decode('utf-8') if prev_state_bytes else "SAFE"
            
        # 4. 状态机逻辑 (滞后阈值)
        # 只要有 3/5 帧是假的，就保持报警；只有降到 1/5 以下才恢复安全
        current_state = prev_state 
        if fake_count >= 3:
            current_state = "ALARM"
        elif fake_count <= 1:
            current_state = "SAFE"
        
        # 更新状态
        if current_state != prev_state:
            r.setex(state_key, 3600, current_state)
            
        return {
            "final_is_fake": (current_state == "ALARM"), 
            "fake_count": fake_count,
            "state": current_state,
            "prev_state": prev_state  # [关键] 返回上一状态用于检测边缘跳变
        }
        
    except Exception as e:
        logger.error(f"Debounce logic failed: {e}")
        return {"final_is_fake": raw_is_fake, "state": "UNKNOWN", "prev_state": "UNKNOWN", "fake_count": -1}


@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_base64: str, user_id: int, call_id: int) -> Dict:
    """音频检测任务"""
    bind_context(user_id=user_id, call_id=call_id)
    # logger.info(f"Task started: Detect audio (Len: {len(audio_base64)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # 1. 获取通话开始时间
                await ensure_call_record_exists(db, call_id, user_id)

                try:
                    audio_bytes = base64.b64decode(audio_base64)
                except Exception as e:
                    return {"status": "error", "message": "Invalid base64"}

                if settings.COLLECT_TRAINING_DATA:
                    await save_raw_data(audio_bytes, user_id, call_id, "audio", "wav")

                self.update_state(state='PROCESSING', meta={'progress': 50})
                
                result = await model_service.predict_voice(audio_bytes)
                is_fake = result.get('is_fake', False)
                confidence = result.get('confidence', 0.0)
                risk_level = result.get('risk_level', 'low')

                # 音频频率较低(3-5秒一次)，通常不需要像视频那样强力的采样优化
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    voice_confidence=confidence,
                    overall_score=confidence * 100,
                    model_version="v1.0"
                )
                db.add(ai_log)
                await db.commit()

                # 通知服务
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
                    payload_control = {
                        "type": "control",
                        "action": "upgrade_level",
                        "target_level": 2,
                        "config": {"video_fps": 30.0, "ui_message": "检测到AI合成语音，请立即挂断！"}
                    }
                    publish_control_command(user_id, payload_control)

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
    """视频检测任务 (包含 双重DB优化逻辑)"""
    bind_context(user_id=user_id, call_id=call_id)
    # logger.info("Task started: Detect video batch")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # 1. 获取通话开始时间 (用于计算 time_offset)
                call_start_time = await ensure_call_record_exists(db, call_id, user_id)
                
                # 计算当前相对秒数 (防止时区问题导致负数，取绝对值)
                now = datetime.now()
                # 兼容 naive 和 aware datetime
                if call_start_time.tzinfo and not now.tzinfo:
                    now = now.replace(tzinfo=call_start_time.tzinfo)
                elif not call_start_time.tzinfo and now.tzinfo:
                    call_start_time = call_start_time.replace(tzinfo=now.tzinfo)
                
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                # 2. 预处理
                try:
                    video_tensor = VideoProcessor.preprocess_batch(frame_data)
                except Exception as e:
                    return {"status": "error", "message": "Preprocessing failed"}

                if settings.COLLECT_TRAINING_DATA:
                    buffer = io.BytesIO()
                    np.save(buffer, video_tensor)
                    await save_raw_data(buffer.getvalue(), user_id, call_id, "video_tensor", "npy")

                # 3. 模型推理
                raw_result = await model_service.predict_video(video_tensor)
                raw_is_fake = raw_result.get('is_deepfake', False)
                raw_conf = raw_result.get('confidence', 0.0)

                # 4. 防抖处理
                debounce_data = apply_video_debounce(user_id, call_id, raw_is_fake)
                
                # 初始化 Redis (供后续两步使用)
                r = redis.from_url(settings.REDIS_URL)

                # =======================================================
                # 5. [DB优化 A] 技术流水日志动态采样 (Dynamic Sampling)
                # =======================================================
                # 策略：高风险(>0.6)每0.5秒记一次；低风险每10秒记一次
                tech_log_key = f"detect:tech_log_throttle:{call_id}"
                log_interval = 0.5 if raw_conf > 0.6 else 10.0
                
                last_tech_ts = r.get(tech_log_key)
                now_ts = now.timestamp()
                should_log_tech = False

                # 如果是第一次，或者距离上次记录超过了间隔，则写入
                if not last_tech_ts or (now_ts - float(last_tech_ts) >= log_interval):
                    should_log_tech = True
                
                if should_log_tech:
                    ai_log = AIDetectionLog(
                        call_id=call_id,
                        video_confidence=raw_conf,
                        overall_score=raw_conf * 100,
                        time_offset=time_offset,
                        model_version=raw_result.get("model_version", "v1.0")
                    )
                    db.add(ai_log)
                    await db.commit()
                    # 更新 Redis 记录时间
                    r.set(tech_log_key, now_ts, ex=3600)

                # =======================================================
                # 6. [DB优化 B] 消息通知边缘触发 (Edge Triggering)
                # =======================================================
                alarm_start_key = f"detect:alarm_start:{call_id}"  # 记录报警开始时间
                last_alert_key = f"detect:last_alert:{call_id}"    # 记录上次通知时间(冷却用)
                
                prev_state = debounce_data.get("prev_state", "SAFE")
                curr_state = debounce_data.get("state", "SAFE")
                
                should_notify = False
                msg_type = "info"
                risk_level = "safe"
                msg_content = ""

                # --- 场景 A: 状态跳变 SAFE -> ALARM (上升沿) ---
                if prev_state == "SAFE" and curr_state == "ALARM":
                    should_notify = True
                    msg_type = "alert"
                    risk_level = "high"
                    msg_content = f"在通话第 {time_offset} 秒检测到疑似伪造内容 (置信度: {raw_conf:.2f})。"
                    
                    # 记录报警开始时刻
                    r.set(alarm_start_key, now_ts)
                    # 重置/设置冷却时间
                    r.set(last_alert_key, now_ts)
                    
                    logger.warning(f"🚨 ALARM START at {time_offset}s (Conf: {raw_conf:.2f})")

                # --- 场景 B: 状态跳变 ALARM -> SAFE (下降沿) ---
                elif prev_state == "ALARM" and curr_state == "SAFE":
                    should_notify = True
                    msg_type = "info"
                    risk_level = "safe"
                    
                    # 计算伪造持续了多久
                    start_ts = r.get(alarm_start_key)
                    duration_str = "未知"
                    if start_ts:
                        duration = int(now_ts - float(start_ts))
                        duration_str = f"{duration}"
                        r.delete(alarm_start_key) # 清除开始时间
                    
                    msg_content = f"环境恢复安全。伪造持续时长: {duration_str}秒 (结束于第 {time_offset} 秒)。"
                    logger.info(f"🟢 ALARM END at {time_offset}s (Duration: {duration_str}s)")

                # --- 场景 C: 持续报警 (ALARM 保持) + 冷却时间 (10s) ---
                elif curr_state == "ALARM":
                    last_alert_ts = r.get(last_alert_key)
                    # 冷却时间设为 10 秒
                    if not last_alert_ts or (now_ts - float(last_alert_ts) > 10.0):
                        should_notify = True
                        msg_type = "alert"
                        risk_level = "high"
                        msg_content = f"当前仍处于伪造状态 (已持续监控中...)"
                        
                        # 更新最后通知时间
                        r.set(last_alert_key, now_ts)
                        logger.info(f"🔁 Sustained Alarm pulse at {time_offset}s")

                # 7. 仅在满足条件时写入 message_logs 并推送到前端
                if should_notify:
                    await notification_service.handle_detection_result(
                        db=db,
                        user_id=user_id,
                        call_id=call_id,
                        detection_type="视频",
                        is_risk=(curr_state == "ALARM"),
                        confidence=raw_conf,
                        risk_level=risk_level,
                        details=msg_content
                    )

                # 8. 触发高危防御 (仅在 ALARM 状态且置信度极高时触发)
                if curr_state == "ALARM" and raw_conf > 0.95:
                     payload_control = {
                        "type": "control",
                        "action": "upgrade_level",
                        "target_level": 2,
                        "config": {"video_fps": 30.0, "ui_message": "检测到AI换脸，请警惕！"}
                    }
                     publish_control_command(user_id, payload_control)

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
    """文本检测任务 (Agent重构版)"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Detect text via Agent (Len: {len(text)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                await ensure_call_record_exists(db, call_id, user_id)
                self.update_state(state='PROCESSING', meta={'progress': 0})
                
                # 1. 规则引擎优先匹配
                rule_hit = None
                try:
                    rule_hit = await security_service.match_risk_rules(text)
                except Exception as e:
                    logger.error(f"Risk rule matching failed: {e}")

                if rule_hit:
                    logger.warning(f"规则匹配: {rule_hit['keyword']}")
                    risk_level_code = rule_hit.get('risk_level', 1)
                    
                    await notification_service.handle_detection_result(
                        db=db, user_id=user_id, call_id=call_id,
                        detection_type="文本(规则)", is_risk=True, confidence=1.0,
                        risk_level="high" if risk_level_code >= 4 else "medium",
                        details=f"触发敏感词: {rule_hit['keyword']}"
                    )
                    return {"status": "success", "result": {"is_fraud": True, "source": "rule"}}

                # 2. 查询用户画像以实现个性化 Prompt
                user_stmt = select(User).where(User.user_id == user_id)
                user_result = await db.execute(user_stmt)
                user = user_result.scalar_one_or_none()
                role_type = user.role_type if user and user.role_type else "青壮年"
                
                self.update_state(state='PROCESSING', meta={'progress': 40})
                
                # 3. 核心：调用 LLM 智能体进行 RAG 检索与意图推理
                llm_result = await llm_service.analyze_text_risk(text, role_type)
                self.update_state(state='PROCESSING', meta={'progress': 80})
                
                is_fraud = llm_result.get('is_fraud', False)
                confidence = llm_result.get('confidence', 0.0)
                risk_level = llm_result.get('risk_level', 'safe')

                # 4. 通知服务
                await notification_service.handle_detection_result(
                    db=db, user_id=user_id, call_id=call_id,
                    detection_type="文本(大模型)", is_risk=is_fraud, confidence=confidence,
                    risk_level=risk_level,
                    details=f"AI意图分析: {llm_result.get('analysis')[:50]}..."
                )

                if is_fraud and risk_level in ['high', 'critical']:
                    # 发送控制指令，可以在这里把 LLM 给出的 advice 展示给前端
                    payload_control = {
                        "type": "control", "action": "upgrade_level", "target_level": 2,
                        "config": {
                            "ui_message": f"智能体预警: {llm_result.get('advice')}",
                            "warning_mode": "modal"
                        }
                    }
                    publish_control_command(user_id, payload_control)

                return {"status": "success", "result": llm_result, "call_id": call_id}
                
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

@celery_app.task(name="multi_modal_fusion", bind=True)
def multi_modal_fusion_task(self, text: str, audio_conf: float, video_conf: float, user_id: int, call_id: int) -> Dict:
    """
    多模态融合决策引擎任务
    接收 ASR 文本、音频鉴伪分数、视频鉴伪分数，交由 LLM 智能体综合裁定
    """
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Multi-Modal Fusion (A:{audio_conf:.2f}, V:{video_conf:.2f}, TextLen:{len(text)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                await ensure_call_record_exists(db, call_id, user_id)
                self.update_state(state='PROCESSING', meta={'progress': 10})
                
                # 1. 优先执行规则匹配
                rule_hit = None
                try:
                    rule_hit = await security_service.match_risk_rules(text)
                except Exception as e:
                    logger.error(f"融合引擎 - 规则匹配异常: {e}")

                if rule_hit:
                    logger.warning(f"融合引擎 - 触发高危强规则: {rule_hit['keyword']}")
                    # 直接触发告警，无需消耗大模型 Token
                    await notification_service.handle_detection_result(
                        db=db, user_id=user_id, call_id=call_id,
                        detection_type="多模态(强规则)", is_risk=True, confidence=1.0,
                        risk_level="critical", details=f"命中强规则: {rule_hit['keyword']}"
                    )
                    return {"status": "success", "result": {"is_fraud": True, "source": "rule"}}

                # 2. 查询用户画像，为 LLM 提供个性化参数
                user_stmt = select(User).where(User.user_id == user_id)
                user_result = await db.execute(user_stmt)
                user = user_result.scalar_one_or_none()
                role_type = user.role_type if user and user.role_type else "青壮年"
                
                self.update_state(state='PROCESSING', meta={'progress': 40})
                
                # 3. 核心：调用 LLM 大模型进行多模态融合裁定
                llm_result = await llm_service.analyze_multimodal_risk(
                    user_input=text, 
                    role_type=role_type, 
                    audio_conf=audio_conf, 
                    video_conf=video_conf
                )
                self.update_state(state='PROCESSING', meta={'progress': 80})
                
                is_fraud = llm_result.get('is_fraud', False)
                risk_level = llm_result.get('risk_level', 'safe')

                # 4. 记录最终决策日志并通知前端
                await notification_service.handle_detection_result(
                    db=db, user_id=user_id, call_id=call_id,
                    detection_type="多模态(Agent融合)", is_risk=is_fraud, 
                    confidence=llm_result.get('confidence', 0.0),
                    risk_level=risk_level,
                    details=f"Agent裁定: {llm_result.get('analysis')[:60]}..."
                )

                # 5. 如果判定为中高危，发布 websocket 控制指令到前端 APP
                if is_fraud and risk_level in ['suspicious', 'fake']:
                    payload_control = {
                        "type": "control", 
                        "action": "upgrade_level", 
                        "target_level": 2 if risk_level == 'fake' else 1,
                        "config": {
                            "ui_message": f"智能防诈守卫: {llm_result.get('advice')}",
                            "warning_mode": "modal",
                            "block_call": (risk_level == 'fake')
                        }
                    }
                    publish_control_command(user_id, payload_control)

                return {"status": "success", "result": llm_result, "call_id": call_id}
                
            except Exception as e:
                logger.error(f"多模态融合任务彻底崩溃: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "call_id": call_id}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_process())
    finally:
        loop.close()