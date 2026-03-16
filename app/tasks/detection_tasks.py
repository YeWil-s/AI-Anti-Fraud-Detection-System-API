"""
AI检测异步任务 (Celery Worker) 
- 采用黑板模式(Blackboard)与文本意图驱动(Semantic-Driven)架构
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
from asgiref.sync import async_to_sync

from sqlalchemy import select
from app.models.call_record import CallRecord, DetectionResult

from app.services.memory_service import memory_service
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

def publish_realtime_score(user_id: int, detection_type: str, is_risk: bool, confidence: float):
    """专门向前端 WebSocket 实时推送分数"""
    try:
        payload = {
            "type": "detection_result",
            "detection_type": detection_type,
            "is_risk": is_risk,
            "confidence": confidence,
            "message": "安全" if not is_risk else "检测到风险"
        }
        message_data = {"user_id": user_id, "payload": payload}
        # 通过 Redis pub/sub 发送给 websocket_manager
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish real-time score: {e}")

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

async def ensure_call_record_exists(db, call_id: int, user_id: int) -> Optional[CallRecord]:
    """
    确保 CallRecord 存在，并返回通话记录对象
    """
    try:
        result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
        record = result.scalar_one_or_none()
        
        if record:
            return record
        
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
        return new_record
    except Exception as e:
        logger.error(f"Failed to ensure call record: {e}")
        # 【修复】原代码这里返回了 datetime.now() 导致后续获取 record.audio_url 时崩溃，现统一返回 None
        return None


# [核心辅助函数] 发布控制指令
def publish_control_command(user_id: int, payload: dict):
    """发送控制指令 (如升级防御等级、挂断通话)"""
    try:
        message_data = {"user_id": user_id, "payload": payload}
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish control command: {e}")


# [核心辅助函数] 视频结果防抖与状态机逻辑
def apply_video_debounce(user_id: int, call_id: int, raw_is_fake: bool) -> dict:
    """使用 Redis 实现滑动窗口防抖和状态机"""
    try:
        r = redis.from_url(settings.REDIS_URL)
        window_key = f"detect:window:{call_id}"
        state_key = f"detect:state:{call_id}"
        
        val = 1 if raw_is_fake else 0
        r.lpush(window_key, val)
        r.ltrim(window_key, 0, 4) 
        r.expire(window_key, 3600)
        
        history = r.lrange(window_key, 0, -1)
        fake_count = sum(int(x) for x in history)
        
        prev_state_bytes = r.get(state_key)
        prev_state = prev_state_bytes.decode('utf-8') if prev_state_bytes else "SAFE"
            
        current_state = prev_state 
        if fake_count >= 3:
            current_state = "ALARM"
        elif fake_count <= 1:
            current_state = "SAFE"
        
        if current_state != prev_state:
            r.setex(state_key, 3600, current_state)
            
        return {
            "final_is_fake": (current_state == "ALARM"), 
            "fake_count": fake_count,
            "state": current_state,
            "prev_state": prev_state
        }
    except Exception as e:
        logger.error(f"Debounce logic failed: {e}")
        return {"final_is_fake": raw_is_fake, "state": "UNKNOWN", "prev_state": "UNKNOWN", "fake_count": -1}


@celery_app.task(name="detect_audio", bind=True)
def detect_audio_task(self, audio_base64: str, user_id: int, call_id: int) -> Dict:
    """音频检测任务 (底层雷达)"""
    bind_context(user_id=user_id, call_id=call_id)

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # 1. 获取通话记录对象
                record = await ensure_call_record_exists(db, call_id, user_id)

                try:
                    audio_bytes = base64.b64decode(audio_base64)
                except Exception as e:
                    return {"status": "error", "message": "Invalid base64"}

                if settings.COLLECT_TRAINING_DATA:
                    await save_raw_data(audio_bytes, user_id, call_id, "audio", "wav")
                
                # 模型预测
                result = await model_service.predict_voice(audio_bytes)
                is_fake = result.get('is_fake', False)
                confidence = result.get('confidence', 0.0)
                #publish_realtime_score(user_id, "语音", is_fake, confidence)
                risk_level = result.get('risk_level', 'low')
                
                # 2. 正确计算 time_offset
                now = datetime.now()
                # 【修复】兜底容错：如果 record 为 None 或者没有 start_time，使用当前时间
                call_start_time = record.start_time if record and hasattr(record, 'start_time') and record.start_time else now
                
                # 时区对齐
                if call_start_time.tzinfo and not now.tzinfo:
                    now = now.replace(tzinfo=call_start_time.tzinfo)
                elif not call_start_time.tzinfo and now.tzinfo:
                    call_start_time = call_start_time.replace(tzinfo=now.tzinfo)
                
                # 相减得到秒数偏移量
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                # 3. 将最新的音频分数写入 Redis "黑板"
                # （注：在 async 函数内使用同步 redis.from_url 建议未来优化为 redis.asyncio）
                r = redis.from_url(settings.REDIS_URL)
                r.setex(f"call:{call_id}:latest_audio_conf", 3600, str(confidence))

                evidence_url = ""
                if confidence > settings.VOICE_DETECTION_THRESHOLD:
                    evidence_url = await save_audit_evidence(audio_bytes, user_id, call_id, "audio", "wav")
                
                # 4. 记录到底层 AI 审计日志
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    voice_confidence=confidence,
                    overall_score=confidence * 100,
                    evidence_snapshot=evidence_url,
                    time_offset=time_offset,
                    model_version="v1.0"
                )
                db.add(ai_log)
                await db.commit()

                # 更新通话记录的留存证据URL
                if evidence_url:
                    if record and not record.audio_url:
                        record.audio_url = evidence_url
                        await db.commit()
                
                # 触发消息告警系统
                audio_alert_level = "safe"
                if is_fake:
                    if confidence >= 0.95:
                        audio_alert_level = "critical"  # 极度确信是AI伪造，致命风险
                    elif confidence >= 0.85:
                        audio_alert_level = "high"      # 高度确信，高风险
                    elif confidence >= 0.75:
                        audio_alert_level = "medium"    # 有一定嫌疑，中风险
                    else:
                        audio_alert_level = "low"       # 嫌疑较小，低风险提示

                # 2. 触发消息告警系统
                await notification_service.handle_detection_result(
                    db=db, user_id=user_id, call_id=call_id,
                    detection_type="语音", is_risk=is_fake, confidence=confidence,
                    risk_level=audio_alert_level,  
                    details=f"检测结果: {'AI合成语音' if is_fake else '真实'} (置信度: {confidence:.2f})"
                )

                # 更新大盘风险状态
                if is_fake and confidence > 0.90:
                    if record and record.detected_result not in [DetectionResult.FAKE, DetectionResult.SUSPICIOUS]:
                        record.detected_result = DetectionResult.SUSPICIOUS
                        await db.commit()

                # 极高风险才越权直接发起强制拦截指令
                if is_fake and confidence >= 0.90:
                    target_lvl = 3 if confidence >= 0.95 else 2
                    ui_msg = "检测到极高风险AI合成语音，强烈建议挂断！" if target_lvl == 3 else "检测到可疑语音特征，已提高警惕！"
                    warning_md = "fullscreen" if target_lvl == 3 else "modal"
                    payload_control = {
                        "type": "control", 
                        "action": "upgrade_level", 
                        "target_level": target_lvl,
                        "config": {
                            "ui_message": ui_msg,
                            "warning_mode": warning_md
                        } 
                    }
                    publish_control_command(user_id, payload_control)

                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Audio task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    try:
        return async_to_sync(_process)()
    except Exception as e:
        logger.error(f"Task wrapper failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(name="detect_video", bind=True)
def detect_video_task(self, frame_data: list, user_id: int, call_id: int) -> Dict:
    """视频检测任务"""
    bind_context(user_id=user_id, call_id=call_id)

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                record = await ensure_call_record_exists(db, call_id, user_id)
                now = datetime.now()
                call_start_time = record.start_time if record and hasattr(record, 'start_time') and record.start_time else now
                
                if call_start_time.tzinfo and not now.tzinfo:
                    now = now.replace(tzinfo=call_start_time.tzinfo)
                elif not call_start_time.tzinfo and now.tzinfo:
                    call_start_time = call_start_time.replace(tzinfo=now.tzinfo)
                
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                try:
                    video_tensor = VideoProcessor.preprocess_batch(frame_data)
                except Exception as e:
                    return {"status": "error", "message": "Preprocessing failed"}

                raw_result = await model_service.predict_video(video_tensor)
                raw_is_fake = raw_result.get('is_deepfake', False)
                raw_conf = raw_result.get('confidence', 0.0)
                #publish_realtime_score(user_id, "视频", raw_is_fake, raw_conf)

                evidence_url = ""
                if raw_conf > settings.VIDEO_DETECTION_THRESHOLD:
                    try:
                        first_frame_bytes = base64.b64decode(frame_data[0])
                        evidence_url = await save_audit_evidence(first_frame_bytes, user_id, call_id, "video", "jpg")
                    except Exception as e:
                        pass

                debounce_data = apply_video_debounce(user_id, call_id, raw_is_fake)
                curr_state = debounce_data.get("state", "SAFE")

                r = redis.from_url(settings.REDIS_URL)
                r.setex(f"call:{call_id}:latest_video_conf", 3600, str(raw_conf))
                r.setex(f"call:{call_id}:latest_video_state", 3600, curr_state)

                tech_log_key = f"detect:tech_log_throttle:{call_id}"
                log_interval = 0.5 if raw_conf > settings.VIDEO_DETECTION_THRESHOLD else 10.0
                last_tech_ts = r.get(tech_log_key)
                now_ts = now.timestamp()
                
                if not last_tech_ts or (now_ts - float(last_tech_ts) >= log_interval):
                    ai_log = AIDetectionLog(
                        call_id=call_id, video_confidence=raw_conf,
                        overall_score=raw_conf * 100, time_offset=time_offset,
                        evidence_snapshot=evidence_url,
                        model_version=raw_result.get("model_version", "v1.0")
                    )
                    db.add(ai_log)
                    await db.commit()
                    r.set(tech_log_key, now_ts, ex=3600)

                if evidence_url:
                    if record and not record.cover_image:
                        record.cover_image = evidence_url
                        await db.commit()
                        
                # 通知边缘触发处理
                alarm_start_key = f"detect:alarm_start:{call_id}"
                last_alert_key = f"detect:last_alert:{call_id}"
                prev_state = debounce_data.get("prev_state", "SAFE")
                
                should_notify = False
                msg_content = ""
                risk_level = "safe"

                if prev_state == "SAFE" and curr_state == "ALARM":
                    should_notify, msg_content, risk_level = True, f"在通话第 {time_offset} 秒检测到疑似伪造内容 (置信度: {raw_conf:.2f})。", "high"
                    r.set(alarm_start_key, now_ts)
                    r.set(last_alert_key, now_ts)
                elif prev_state == "ALARM" and curr_state == "SAFE":
                    should_notify, risk_level = True, "safe"
                    start_ts = r.get(alarm_start_key)
                    duration_str = str(int(now_ts - float(start_ts))) if start_ts else "未知"
                    msg_content = f"环境恢复安全。伪造持续时长: {duration_str}秒 (结束于第 {time_offset} 秒)。"
                    r.delete(alarm_start_key)
                elif curr_state == "ALARM":
                    last_alert_ts = r.get(last_alert_key)
                    if not last_alert_ts or (now_ts - float(last_alert_ts) > 10.0):
                        should_notify, msg_content, risk_level = True, "当前仍处于伪造状态 (已持续监控中...)", "high"
                        r.set(last_alert_key, now_ts)

                if should_notify:
                    await notification_service.handle_detection_result(
                        db=db, user_id=user_id, call_id=call_id, detection_type="视频",
                        is_risk=(curr_state == "ALARM"), confidence=raw_conf,
                        risk_level=risk_level, details=msg_content
                    )

                if curr_state == "ALARM" and raw_conf > 0.80:
                    if record and record.detected_result not in [DetectionResult.FAKE, DetectionResult.SUSPICIOUS]:
                        record.detected_result = DetectionResult.SUSPICIOUS
                        await db.commit()

                if curr_state == "ALARM" and raw_conf >= 0.75:
                    target_lvl = 3 if raw_conf >= 0.90 else 2
                    ui_msg = "画面出现严重篡改痕迹，极高风险！" if target_lvl == 3 else "画面出现可疑篡改痕迹，已提高防御等级！"
                    warning_md = "fullscreen" if target_lvl == 3 else "modal"
                    fps_val = 30.0 if target_lvl == 3 else 15.0
                    payload_control = {
                        "type": "control", 
                        "action": "upgrade_level", 
                        "target_level": target_lvl,
                        "config": {
                            "video_fps": fps_val, 
                            "ui_message": ui_msg,
                            "warning_mode": warning_md
                        } 
                    }
                    publish_control_command(user_id, payload_control)
                    
                return {"status": "success", "result": raw_result, "debounce": debounce_data}
            except Exception as e:
                logger.error(f"Video task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    try:
        return async_to_sync(_process)()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 多模态融合
@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """文本触发器 & 大模型多模态融合决策司令部"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: Text/Fusion Command Center (Len: {len(text)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                # 【修复】直接使用这里获取到的 record，不重复执行 select 语句
                record = await ensure_call_record_exists(db, call_id, user_id)
                
                # 0. 过滤无意义短句 
                if len(text.strip()) < 2:
                    return {"status": "skipped", "reason": "text too short"}
                    
                await memory_service.add_message(call_id, text)

                now = datetime.now()
                call_start_time = record.start_time if record and hasattr(record, 'start_time') and record.start_time else now
                if call_start_time.tzinfo and not now.tzinfo:
                    now = now.replace(tzinfo=call_start_time.tzinfo)
                elif not call_start_time.tzinfo and now.tzinfo:
                    call_start_time = call_start_time.replace(tzinfo=now.tzinfo)
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                # 1. 第一道防线
                rule_hit = None
                try:
                    rule_hit = await security_service.match_risk_rules(text)
                except Exception as e:
                    logger.error(f"Risk rule matching failed: {e}")

                force_llm = False

                if rule_hit:
                    logger.warning(f"触发规则拦截: {rule_hit['keyword']}")
                    risk_level_code = rule_hit.get('risk_level', 1)
                    if risk_level_code >= 4:
                        # 命中诈骗，拉高防御等级
                        publish_control_command(user_id, {
                            "type": "control", "action": "upgrade_level", "target_level": 2,
                            "config": {"ui_message": f"触发高危防骗规则: {rule_hit['keyword']}", "block_call": True}
                        })
                        await notification_service.handle_detection_result(
                            db=db, user_id=user_id, call_id=call_id,
                            detection_type="多模态(强规则)", is_risk=True, confidence=1.0,
                            risk_level="critical", details=f"命中强规则: {rule_hit['keyword']}"
                        )
                        force_llm = True

                try:
                    text_result = await model_service.predict_text(text)
                    text_conf = text_result.get("confidence", 0.5) 
                    logger.info(f"本地 ONNX 文本检测置信度: {text_conf:.4f}")
                    
                    if force_llm:
                        logger.info("已命中高危风险词，跳过 ONNX 安全拦截，强制交由大模型裁决")
                        text_conf = max(text_conf, 0.95) 
                    elif text_conf < 0.3:
                        # 极速拦截：如果是极度安全的日常闲聊 (低于0.3)
                        logger.info("ONNX 判定为安全闲聊，跳过大模型处理")
                        ai_log = AIDetectionLog(
                            call_id=call_id,
                            text_confidence=text_conf,      # 极低风险分数
                            overall_score=text_conf * 100,
                            detected_keywords=text,         # 存入原文
                            time_offset=time_offset,
                            model_version="onnx-fast"       # 标记来源为轻量模型
                        )
                        db.add(ai_log)
                        await db.commit()
                        #publish_realtime_score(user_id, "文本", False, text_conf)
                        return {"status": "success", "result": {"is_fraud": False, "risk_level": "safe", "source": "onnx"}}

                except Exception as e:
                    logger.error(f"ONNX 文本快筛异常: {e}")
                    text_conf = 0.95 if force_llm else 0.5 # 如果崩了，给个中性分数

                # 2. 从数据库动态获取通话场景
                platform_raw = getattr(record, 'platform', 'PHONE') if record else 'PHONE'
                platform_str = platform_raw.name if hasattr(platform_raw, 'name') else str(platform_raw)
                platform_str_lower = platform_str.lower()
                is_video_call = "video" in platform_str_lower or "视频" in platform_str_lower

                # 3. 组装多维度用户画像
                user_stmt = select(User).where(User.user_id == user_id)
                user_result = await db.execute(user_stmt)
                user = user_result.scalar_one_or_none()
                
                profile_parts = []
                if user:
                    if getattr(user, 'role_type', None): profile_parts.append(f"身份：{user.role_type}")
                    if getattr(user, 'gender', None): profile_parts.append(f"性别：{user.gender}")
                    if getattr(user, 'profession', None): profile_parts.append(f"职业：{user.profession}")
                    if getattr(user, 'marital_status', None): profile_parts.append(f"婚姻：{user.marital_status}")
                user_profile_str = "，".join(profile_parts) if profile_parts else "普通青壮年"

                # 4. 从 Redis "黑板" 拉取最新底层音视频得分
                r = redis.from_url(settings.REDIS_URL)
                audio_conf = float(r.get(f"call:{call_id}:latest_audio_conf") or 0.0)
                
                if is_video_call:
                    video_conf_val = f"{float(r.get(f'call:{call_id}:latest_video_conf') or 0.0):.4f}"
                    call_type_desc = f"视频通话场景（当前渠道：{platform_str}。请综合分析画面伪造与语音伪造特征）"
                else:
                    video_conf_val = "N/A"
                    call_type_desc = f"纯语音通话场景（当前渠道：{platform_str}。请完全忽略视频特征，重点分析文本与语音）"

                # 5. 维护短期记忆池
                
                chat_history = await memory_service.get_context(call_id)
                
                # 6. 第三道防线：呼叫 LLM 大脑融合裁决
                llm_result = await llm_service.analyze_multimodal_risk(
                    user_input=text, 
                    chat_history=chat_history,
                    user_profile=user_profile_str,
                    call_type=call_type_desc,
                    audio_conf=audio_conf,
                    video_conf=video_conf_val,
                    text_conf=text_conf  
                )
                
                is_fraud = llm_result.get('is_fraud', False)
                llm_conf = float(llm_result.get('confidence', text_conf))

                ai_log = AIDetectionLog(
                    call_id=call_id,
                    text_confidence=llm_conf,    
                    overall_score=llm_conf * 100,
                    detected_keywords=text,        # 存入原文
                    time_offset=time_offset,
                    model_version="llm-fusion"     # 标记来源为大模型
                )
                db.add(ai_log)
                await db.commit()

                # 【修复】删除了此处重复冗余的 publish_realtime_score 调用
                risk_level = llm_result.get('risk_level', 'safe')
                raw_llm_risk = risk_level.lower()

                # A. 映射到 MessageLog (告警级别: safe, low, medium, high, critical)
                alert_level = 'safe'
                if is_fraud:
                    if raw_llm_risk in ['critical', 'fake']:
                        alert_level = 'critical'
                    elif raw_llm_risk in ['high']:
                        alert_level = 'high'
                    elif raw_llm_risk in ['suspicious', 'medium']:
                        alert_level = 'medium'
                    else:
                        alert_level = 'low'
                
                record_verdict = DetectionResult.SAFE
                if is_fraud:
                    if raw_llm_risk in ['critical', 'fake', 'high']:
                        record_verdict = DetectionResult.FAKE
                    else:
                        record_verdict = DetectionResult.SUSPICIOUS
                
                target_level = 1
                if is_fraud:
                    target_level = 3 if record_verdict == DetectionResult.FAKE else 2

                #publish_realtime_score(user_id, "文本", is_fraud, llm_result.get('confidence', 0.0))
                full_analysis = llm_result.get('analysis', '')
                full_advice = llm_result.get('advice', '')

                # 7. 写入最终决策日志并通知前端
                await notification_service.handle_detection_result(
                    db=db, user_id=user_id, call_id=call_id,
                    detection_type="多模态(Agent融合)", is_risk=is_fraud, 
                    confidence=llm_result.get('confidence', 0.0),
                    risk_level=alert_level,
                    details=f"Agent裁定: {full_analysis}"
                )

                if record:
                    # 无论是否诈骗，只要大模型给了评价，我们就更新最新评价
                    record.analysis = full_analysis
                    record.advice = full_advice
                    
                    current_verdict = record.detected_result
                    if record_verdict == DetectionResult.FAKE and current_verdict != DetectionResult.FAKE:
                        record.detected_result = DetectionResult.FAKE
                    elif record_verdict == DetectionResult.SUSPICIOUS and current_verdict == DetectionResult.SAFE:
                        record.detected_result = DetectionResult.SUSPICIOUS
                        
                    await db.commit()

                # 8. LLM 下达 WebSocket 控制指令 (动态升降级)
                if target_level > 1:
                    payload_control = {
                        "type": "control", "action": "upgrade_level", "target_level": target_level,
                        "config": {
                            "video_fps": 15.0 if target_level == 2 else 30.0,
                            "ui_message": f"智能防诈守卫: {full_advice}",
                            "warning_mode": "modal" if target_level == 2 else "fullscreen",
                            "block_call": (target_level == 3)
                        }
                    }
                    publish_control_command(user_id, payload_control)

                return {"status": "success", "result": llm_result, "call_id": call_id}
                
            except Exception as e:
                logger.error(f"Text fusion command center failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "call_id": call_id}

    try:
        return async_to_sync(_process)()
    except Exception as e:
        logger.error(f"Task wrapper failed: {e}")
        return {"status": "error", "message": str(e), "call_id": call_id}

@celery_app.task(name="get_task_status")
def get_task_status(task_id: str) -> Dict:
    res = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": res.status, "result": res.result if res.successful() else None}

# 新增保存审计证据的函数，返回文件的 MinIO URL
async def save_audit_evidence(data: bytes, user_id: int, call_id: int, data_type: str, ext: str) -> str:
    """保存审计证据到 MinIO 并返回访问路径"""
    try:
        timestamp = int(time.time() * 1000)
        filename = f"audit/{data_type}/{user_id}/{call_id}_{timestamp}.{ext}"
        content_type = "audio/wav" if data_type == "audio" else "image/jpeg"
        
        file_url = await upload_to_minio(data, filename, content_type=content_type)
        return file_url
    except Exception as e:
        logger.error(f"审计证据保存失败: {e}")
        return ""

@celery_app.task(name="generate_post_call_summary")
def generate_post_call_summary_task(call_id: int, user_id: int):
    """通话结束后，由 Celery 去读取自身内存中的对话并呼叫 LLM"""
    
    async def _process():
        # 1. 在 Celery 进程内读取 chat_history (此时它是有数据的)
        chat_history = await memory_service.get_context(call_id)
        
        # 过滤空对话：如果真的没有说话，为了省钱直接清空并跳过
        if not chat_history or chat_history == "暂无历史上下文。" or len(chat_history) == 0:
            await memory_service.clear_context(call_id)
            return {"status": "skipped", "reason": "No conversation history"}
            
        try:
            # 2. 呼叫大模型进行全盘总结
            summary_result = await llm_service.generate_final_summary(chat_history)
            
            # 3. 开启数据库会话进行更新
            async with AsyncSessionLocal() as db:
                result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
                record = result.scalar_one_or_none()
                
                if record:
                    record.analysis = summary_result.get("analysis", record.analysis)
                    record.advice = summary_result.get("advice", record.advice)
                    
                    final_risk = summary_result.get("risk_level", "safe").lower()
                    
                    record_verdict = DetectionResult.SAFE
                    if final_risk in ['fake', 'high', 'critical']:
                        record_verdict = DetectionResult.FAKE
                    elif final_risk in ['suspicious', 'medium']:
                        record_verdict = DetectionResult.SUSPICIOUS

                    record.detected_result = record_verdict
                    await db.commit()
                    
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            # 4. 无论成功失败，处理完毕后清空该通话的内存，防止内存泄漏
            memory_service.clear_context(call_id)

    # 运行异步逻辑
    try:
        return async_to_sync(_process)()
    except Exception as e:
        return {"status": "error", "message": str(e)}