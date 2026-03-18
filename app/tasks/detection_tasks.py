"""
AI检测异步任务 
- 采用黑板模式与文本意图驱动架构
- 引入马尔可夫决策过程实现动态防御阈值调整
"""
import cv2
import tempfile
import os
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
from app.services.risk_fusion_engine import fusion_engine
from app.services.mdp_defense.dynamic_defense_agent import DynamicDefenseAgent
from app.services.image_ocr_service import image_ocr_service
from app.core.redis import get_redis

# 初始化模块级 logger
logger = get_logger(__name__)

# 全局初始化 MDP 智能体
mdp_agent = DynamicDefenseAgent()

# Redis 连接池（模块级复用）
_redis_pool = None

def get_redis_pool():
    """获取 Redis 连接池"""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_pool

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
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish real-time score: {e}")

async def save_raw_data(data: bytes, user_id: int, call_id: int, data_type: str, ext: str):
    """保存原始数据到 MinIO"""
    if not getattr(settings, 'COLLECT_TRAINING_DATA', False):
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
        return None

def publish_control_command(user_id: int, payload: dict):
    """发送控制指令 (如升级防御等级、挂断通话)"""
    try:
        message_data = {"user_id": user_id, "payload": payload}
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
    except Exception as e:
        logger.error(f"Failed to publish control command: {e}")

def apply_video_debounce(user_id: int, call_id: int, raw_is_fake: bool) -> dict:
    """使用 Redis 实现滑动窗口防抖和状态机"""
    try:
        r = get_redis_pool()
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
                record = await ensure_call_record_exists(db, call_id, user_id)
                try:
                    audio_bytes = base64.b64decode(audio_base64)
                except Exception as e:
                    return {"status": "error", "message": "Invalid base64"}

                if getattr(settings, 'COLLECT_TRAINING_DATA', False):
                    await save_raw_data(audio_bytes, user_id, call_id, "audio", "wav")
                
                result = await model_service.predict_voice(audio_bytes)
                is_fake = result.get('is_fake', False)
                confidence = result.get('confidence', 0.0)
                
                now = datetime.now()
                call_start_time = record.start_time if record and hasattr(record, 'start_time') and record.start_time else now
                
                if call_start_time.tzinfo and not now.tzinfo:
                    now = now.replace(tzinfo=call_start_time.tzinfo)
                elif not call_start_time.tzinfo and now.tzinfo:
                    call_start_time = call_start_time.replace(tzinfo=now.tzinfo)
                
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                r = get_redis_pool()
                r.setex(f"call:{call_id}:latest_audio_conf", 3600, str(confidence))

                evidence_url = ""
                if confidence > getattr(settings, 'VOICE_DETECTION_THRESHOLD', 0.8):
                    evidence_url = await save_audit_evidence(audio_bytes, user_id, call_id, "audio", "wav")
                
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

                if evidence_url and record and not record.audio_url:
                    record.audio_url = evidence_url
                    await db.commit()
                
                # 音频高危特征：提升防御等级，但不直接触发警报
                # 最终决策交给文本检测融合判断
                if is_fake and confidence >= 0.95:
                    # 只提升防御等级，增加检测频率
                    payload_control = {
                        "type": "control", "action": "upgrade_level", "target_level": 2,
                        "config": {
                            "ui_message": "【风险提示】检测到AI合成语音特征，请提高警惕", 
                            "warning_mode": "normal",
                            "video_fps": 15.0  # 增加检测频率
                        } 
                    }
                    publish_control_command(user_id, payload_control)
                    
                    # 记录到Redis，供文本检测融合时参考
                    r = get_redis_pool()
                    r.setex(f"call:{call_id}:audio_high_risk_flag", 300, "1")
                    
                    logger.warning(f"音频检测到高危特征，已提升防御等级，等待文本融合判断 | 置信度: {confidence:.2f}")
                
                return {"status": "success", "result": result}
            except Exception as e:
                logger.error(f"Audio task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    try:
        return async_to_sync(_process)()
    except Exception as e:
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
                
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                video_tensor = VideoProcessor.preprocess_batch(frame_data)
                raw_result = await model_service.predict_video(video_tensor)
                raw_is_fake = raw_result.get('is_deepfake', False)
                raw_conf = raw_result.get('confidence', 0.0)

                evidence_url = ""
                if raw_conf > getattr(settings, 'VIDEO_DETECTION_THRESHOLD', 0.8):
                    try:
                        # 1. 把传入的 base64 列表转回 OpenCV 图片格式
                        frames = []
                        for b64_str in frame_data:
                            img_bytes = base64.b64decode(b64_str)
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                frames.append(frame)

                        # 2. 如果有画面，将其合成为 MP4
                        if frames:
                            height, width, _ = frames[0].shape
                            
                            # 生成一个临时文件路径
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                                temp_video_path = temp_video.name

                            # 初始化视频写入器 (mp4v 编码，假设前端传过来相当于 10 帧/秒)
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(temp_video_path, fourcc, 10.0, (width, height))

                            # 依次写入每一帧
                            for frame in frames:
                                video_writer.write(frame)
                            
                            video_writer.release() # 必须 release 才能保存完整

                            # 3. 读取成字节流准备上传
                            with open(temp_video_path, 'rb') as f:
                                video_bytes = f.read()

                            # 清理本地临时文件
                            os.remove(temp_video_path)

                            # 4. 调用新写的方法，上传生成的短视频！
                            evidence_url = await save_video_evidence(video_bytes, user_id, call_id)
                    except Exception as e:
                        logger.error(f"合成高危短视频失败: {e}", exc_info=True)

                debounce_data = apply_video_debounce(user_id, call_id, raw_is_fake)
                curr_state = debounce_data.get("state", "SAFE")

                r = get_redis_pool()
                r.setex(f"call:{call_id}:latest_video_conf", 3600, str(raw_conf))
                r.setex(f"call:{call_id}:latest_video_state", 3600, curr_state)

                tech_log_key = f"detect:tech_log_throttle:{call_id}"
                log_interval = 0.5 if raw_conf > 0.8 else 10.0
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

                # 视频高危特征：提升防御等级，但不直接触发警报
                # 最终决策交给文本检测融合判断
                if curr_state == "ALARM" and raw_conf >= 0.95:
                    # 只提升防御等级，增加检测频率
                    payload_control = {
                        "type": "control", "action": "upgrade_level", "target_level": 2,
                        "config": {
                            "video_fps": 30.0, 
                            "ui_message": "【风险提示】画面存在异常，请提高警惕", 
                            "warning_mode": "normal"
                        } 
                    }
                    publish_control_command(user_id, payload_control)
                    
                    # 记录到Redis，供文本检测融合时参考
                    r.setex(f"call:{call_id}:video_high_risk_flag", 300, "1")
                    
                    logger.warning(f"视频检测到高危特征，已提升防御等级，等待文本融合判断 | 置信度: {raw_conf:.2f}")
                    
                return {"status": "success", "result": raw_result, "debounce": debounce_data}
            except Exception as e:
                logger.error(f"Video task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}

    try:
        return async_to_sync(_process)()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 核心中枢：多模态融合与 MDP 决策
@celery_app.task(name="detect_text", bind=True)
def detect_text_task(self, text: str, user_id: int, call_id: int) -> Dict:
    """文本触发器 & 大模型多模态融合决策司令部"""
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: MDP Fusion Command Center (Len: {len(text)})")

    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                record = await ensure_call_record_exists(db, call_id, user_id)
                
                if len(text.strip()) < 2:
                    return {"status": "skipped", "reason": "text too short"}
                    
                await memory_service.add_message(call_id, text)

                now = datetime.now()
                call_start_time = record.start_time if record and hasattr(record, 'start_time') and record.start_time else now
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0

                # 1. 第一道防线：强规则库匹配
                rule_hit = await security_service.match_risk_rules(text)
                force_llm = False

                if rule_hit:
                    logger.warning(f"触发规则拦截: {rule_hit['keyword']}")
                    risk_level_code = rule_hit.get('risk_level', 1)
                    if risk_level_code >= 4:
                        publish_control_command(user_id, {
                            "type": "control", "action": "upgrade_level", "target_level": 3,
                            "config": {"ui_message": f"触发高危防骗规则: {rule_hit['keyword']}", "block_call": True}
                        })
                        force_llm = True

                try:
                    text_result = await model_service.predict_text(text)
                    text_conf = text_result.get("confidence", 0.5) 
                    
                    if force_llm:
                        text_conf = max(text_conf, 0.95) 
                    elif text_conf < 0.3:
                        logger.info("ONNX 判定为安全闲聊，跳过重量级决策")
                        return {"status": "success", "result": {"is_fraud": False, "risk_level": "safe", "source": "onnx"}}
                except Exception as e:
                    logger.error(f"ONNX 文本快筛异常: {e}")
                    text_conf = 0.95 if force_llm else 0.5 

                # 2. 准备状态机所需的上下文数据
                platform_raw = getattr(record, 'platform', 'PHONE') if record else 'PHONE'
                platform_str = platform_raw.name if hasattr(platform_raw, 'name') else str(platform_raw)
                is_video_call = "video" in platform_str.lower() or "视频" in platform_str.lower()

                # 用户画像
                user_stmt = select(User).where(User.user_id == user_id)
                user_result = await db.execute(user_stmt)
                user = user_result.scalar_one_or_none()
                
                profile_parts = []
                if user:
                    if getattr(user, 'role_type', None): profile_parts.append(f"身份：{user.role_type}")
                    if getattr(user, 'profession', None): profile_parts.append(f"职业：{user.profession}")
                user_profile_str = "，".join(profile_parts) if profile_parts else "普通人群"

                # 从黑板拉取多模态数据
                r = get_redis_pool()
                audio_conf = float(r.get(f"call:{call_id}:latest_audio_conf") or 0.0)
                raw_video_conf = float(r.get(f"call:{call_id}:latest_video_conf") or 0.0) if is_video_call else 0.0
                
                # 检查音视频高危标志（由音视频检测设置）
                audio_high_risk = r.get(f"call:{call_id}:audio_high_risk_flag") == "1"
                video_high_risk = r.get(f"call:{call_id}:video_high_risk_flag") == "1"
                
                # 如果有音视频高危标志，提升本地文本置信度
                if audio_high_risk and text_conf < 0.5:
                    text_conf = max(text_conf, 0.5)
                    logger.warning(f"音频高危标志触发，提升文本置信度至 {text_conf:.2f}")
                
                if video_high_risk and text_conf < 0.5:
                    text_conf = max(text_conf, 0.5)
                    logger.warning(f"视频高危标志触发，提升文本置信度至 {text_conf:.2f}")
                
                call_type_desc = f"视频通话场景" if is_video_call else f"纯语音通话场景"

                # 提取交互轮次
                chat_history = await memory_service.get_context(call_id)
                msg_count = len(chat_history.split('\n')) if isinstance(chat_history, str) else 1

                # 3. LLM 感知层 (只负责输出分类和意图)
                llm_result = await llm_service.analyze_multimodal_risk(
                    user_input=text, 
                    chat_history=chat_history,
                    user_profile=user_profile_str,
                    call_type=call_type_desc,
                    audio_conf=audio_conf,
                    video_conf=f"{raw_video_conf:.4f}" if is_video_call else "N/A",
                    text_conf=text_conf  
                )
                
                # 4. 融合算分层
                fused_score = fusion_engine.calculate_score(
                    llm_classification=llm_result,
                    local_text_conf=text_conf,
                    audio_conf=audio_conf,
                    video_conf=raw_video_conf
                )

                # 5. 马尔可夫决策层
                # 根据画像、计算的科学分数、交互轮次获取 O(1) 查表动作
                action_level = mdp_agent.get_defense_action(
                    user=user, 
                    current_risk_score=fused_score, 
                    message_count=msg_count
                )

                # 将日志数据落盘
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    text_confidence=fused_score / 100.0, # 兼容老数据库格式 
                    overall_score=fused_score,           # 得出的分数
                    detected_keywords=f"剧本:{llm_result.get('match_script', '无')}|意图:{llm_result.get('intent', '无')}",
                    time_offset=time_offset,
                    model_version="fusion-mdp-v1"
                )
                db.add(ai_log)
                await db.commit()

                # 6. 转化与执行 MDP 的拦截指令
                alert_level = 'safe'
                record_verdict = DetectionResult.SAFE
                ui_message = "请注意保护个人隐私。"
                
                target_level = action_level + 1 # Action(0,1,2) 对应 System Level(1,2,3)

                if action_level == 2:
                    # MDP 裁定：三级最高防御 (强制阻断 + 联动监护人)
                    alert_level = 'critical'
                    record_verdict = DetectionResult.FAKE
                    ui_message = f"【极度高危】系统已阻断。防骗建议：{llm_result.get('advice', '请立刻停止操作！')}"
                    
                    # 触发联动闭环 - 发送 WebSocket 和邮件通知给监护人
                    await notification_service.handle_detection_result(
                        db=db, user_id=user_id, call_id=call_id,
                        detection_type="MDP三级防御-强制阻断", is_risk=True, 
                        confidence=fused_score / 100.0,
                        risk_level='critical',
                        details=f"触发三级防御：{llm_result.get('match_script', '未知剧本')} | 融合分: {fused_score:.1f} | 建议: {llm_result.get('advice', '')}"
                    )

                elif action_level == 1:
                    # MDP 裁定：二级中度防御 (弹窗警告/强核验)
                    alert_level = 'medium'
                    record_verdict = DetectionResult.SUSPICIOUS
                    ui_message = f"【风险提示】行为可疑。防骗建议：{llm_result.get('advice', '请仔细核实对方身份！')}"
                
                # 通知前端展示并在记录中更新
                await notification_service.handle_detection_result(
                    db=db, user_id=user_id, call_id=call_id,
                    detection_type="MDP动态决策", is_risk=(action_level > 0), 
                    confidence=fused_score / 100.0,
                    risk_level=alert_level,
                    details=f"命中剧本: {llm_result.get('match_script', '无')} | 融合分: {fused_score:.1f}"
                )

                if record:
                    record.analysis = llm_result.get('analysis', '')
                    record.advice = llm_result.get('advice', '')
                    
                    current_verdict = record.detected_result
                    if record_verdict == DetectionResult.FAKE:
                        record.detected_result = DetectionResult.FAKE
                    elif record_verdict == DetectionResult.SUSPICIOUS and current_verdict == DetectionResult.SAFE:
                        record.detected_result = DetectionResult.SUSPICIOUS
                    await db.commit()

                # 将指令通过 WebSocket 路由给前端
                if target_level > 1:
                    payload_control = {
                        "type": "control", "action": "upgrade_level", "target_level": target_level,
                        "config": {
                            "video_fps": 15.0 if target_level == 2 else 30.0,
                            "ui_message": ui_message,
                            "warning_mode": "modal" if target_level == 2 else "fullscreen",
                            "block_call": (target_level == 3)
                        }
                    }
                    publish_control_command(user_id, payload_control)

                return {"status": "success", "fused_score": fused_score, "action_level": action_level}
                
            except Exception as e:
                logger.error(f"MDP Fusion Command Center failed: {e}", exc_info=True)
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

async def save_video_evidence(video_bytes: bytes, user_id: int, call_id: int) -> str:
    """专用于保存高危 MP4 视频证据到 MinIO 并返回访问路径"""
    try:
        timestamp = int(time.time() * 1000)
        filename = f"audit/video/{user_id}/{call_id}_{timestamp}.mp4"
        
        # 明确指定 content_type 为 mp4
        file_url = await upload_to_minio(video_bytes, filename, content_type="video/mp4")
        return file_url
    except Exception as e:
        logger.error(f"视频证据保存失败: {e}")
        return ""

@celery_app.task(name="detect_image", bind=True)
def detect_image_task(self, image_base64: str, user_id: int, call_id: int) -> Dict:
    """
    图片检测任务 - 使用GLM-4V-Flash提取文字，支持去重和增量识别
    """
    bind_context(user_id=user_id, call_id=call_id)
    
    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                record = await ensure_call_record_exists(db, call_id, user_id)
                
                # 解码图片
                try:
                    image_bytes = base64.b64decode(image_base64)
                except Exception as e:
                    return {"status": "error", "message": "Invalid base64 image"}
                
                # 保存原始图片（如果需要）
                if getattr(settings, 'COLLECT_TRAINING_DATA', False):
                    await save_raw_data(image_bytes, user_id, call_id, "image", "jpg")
                
                # 使用OCR提取文字
                logger.info(f"开始图片OCR，用户: {user_id}, 通话: {call_id}")
                ocr_result = await image_ocr_service.extract_chat_from_screenshot(image_bytes)
                
                if not ocr_result or not ocr_result.get("text"):
                    return {"status": "success", "result": {"text": "", "is_fraud": False, "source": "ocr_empty"}}
                
                extracted_text = ocr_result["text"]
                logger.info(f"图片OCR完成，提取文字长度: {len(extracted_text)}")
                
                # 计算对话哈希
                dialogue_hash = image_ocr_service.calculate_dialogue_hash(extracted_text)
                
                # 查询该通话最近10分钟的OCR记录，用于去重
                from sqlalchemy import select, and_, desc
                from datetime import timedelta
                
                ten_minutes_ago = datetime.now() - timedelta(minutes=10)
                recent_logs_result = await db.execute(
                    select(AIDetectionLog)
                    .where(
                        and_(
                            AIDetectionLog.call_id == call_id,
                            AIDetectionLog.detection_type == "image",
                            AIDetectionLog.created_at >= ten_minutes_ago
                        )
                    )
                    .order_by(desc(AIDetectionLog.created_at))
                    .limit(5)
                )
                recent_logs = recent_logs_result.scalars().all()
                
                # 获取之前的对话文本用于增量识别
                previous_texts = [log.image_ocr_text for log in recent_logs if log.image_ocr_text]
                
                # 增量识别：找出新增对话
                increment_result = image_ocr_service.extract_new_dialogues(
                    current_text=extracted_text,
                    previous_texts=previous_texts,
                    similarity_threshold=0.85
                )
                
                # 记录OCR结果到数据库
                now = datetime.now()
                call_start_time = record.start_time if record and hasattr(record, 'start_time') and record.start_time else now
                time_offset = int((now - call_start_time).total_seconds())
                if time_offset < 0: time_offset = 0
                
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    detection_type="image",
                    detected_keywords=f"图片OCR提取: {extracted_text[:100]}...",
                    image_ocr_text=extracted_text,  # 保存完整OCR文本
                    ocr_dialogue_hash=dialogue_hash,  # 保存哈希用于去重
                    overall_score=0,
                    time_offset=time_offset,
                    model_version="glm-4v-flash-ocr"
                )
                db.add(ai_log)
                await db.commit()
                
                # 如果是重复内容，只记录不触发检测
                if increment_result["is_duplicate"]:
                    logger.info(f"图片内容重复，相似度: {increment_result['similarity']:.2%}，跳过检测")
                    return {
                        "status": "success", 
                        "result": {
                            "text": extracted_text,
                            "is_duplicate": True,
                            "similarity": increment_result["similarity"],
                            "source": "ocr_duplicate"
                        }
                    }
                
                # 使用新增内容触发文本检测
                new_content = increment_result["new_content"] or extracted_text
                new_lines = increment_result["new_lines"]
                
                if len(new_content.strip()) > 2:
                    logger.info(f"触发文本检测，新增内容长度: {len(new_content)}, 新增行数: {len(new_lines)}")
                    detect_text_task.delay(new_content, user_id, call_id)
                
                return {
                    "status": "success", 
                    "result": {
                        "text": extracted_text,
                        "new_content": new_content,
                        "new_lines": new_lines,
                        "similarity": increment_result["similarity"],
                        "is_duplicate": False,
                        "dialogue": ocr_result.get("dialogue", []),
                        "source": "ocr"
                    }
                }
                
            except Exception as e:
                logger.error(f"Image task failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e)}
    
    try:
        return async_to_sync(_process)()
    except Exception as e:
        return {"status": "error", "message": str(e)}


@celery_app.task(name="generate_post_call_summary")
def generate_post_call_summary_task(call_id: int, user_id: int):
    """通话结束后，由 Celery 去读取自身内存中的对话并呼叫 LLM 进行复盘"""
    async def _process():
        chat_history = await memory_service.get_context(call_id)
        
        if not chat_history or chat_history == "暂无历史上下文。" or len(chat_history) == 0:
            await memory_service.clear_context(call_id)
            return {"status": "skipped", "reason": "No conversation history"}
            
        try:
            summary_result = await llm_service.generate_final_summary(chat_history)
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
            memory_service.clear_context(call_id)

    try:
        return async_to_sync(_process)()
    except Exception as e:
        return {"status": "error", "message": str(e)}