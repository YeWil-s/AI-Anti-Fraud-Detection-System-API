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
from app.models.call_record import CallRecord, DetectionResult, CallPlatform

from app.services.memory_service import memory_service
from app.services.long_term_memory_service import long_term_memory_service
from app.services.llm_service import llm_service
from app.models.user import User
from app.tasks.celery_app import celery_app
from app.services.model_service import model_service
from app.services.video_processor import VideoProcessor
from app.services.security_service import security_service
from app.services.notification_service import notification_service
from app.db.database import AsyncSessionLocal
from app.db.transaction import TransactionContext
from app.models.ai_detection_log import AIDetectionLog
from app.models.chat_message import ChatMessage
from app.models.mdp_decision_event import MDPDecisionEvent

from app.core.storage import upload_to_minio
from app.core.config import settings
from app.core.logger import get_logger, bind_context
from app.services.risk_fusion_engine import fusion_engine, fusion_engine_v2, environment_fusion_engine
from app.services.mdp_defense.dynamic_defense_agent import DynamicDefenseAgent
from app.services.mdp_defense.mdp_types import DefenseAction, MDPObservation, MDPState
from app.services.mdp_defense.reward_builder import reward_builder
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


def _json_dumps(data) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def _published_defense_redis_key(call_id: int) -> str:
    return f"mdp:published_defense:{call_id}"


def _get_published_defense_level(call_id: int) -> int:
    """本通通话最近一次下发给前端的系统防御等级 1/2/3；无记录时视为 1（隐式监测）。"""
    try:
        r = get_redis_pool()
        v = r.get(_published_defense_redis_key(call_id))
        if v is None:
            return 1
        return int(v)
    except Exception:
        return 1


def _set_published_defense_level(call_id: int, level: int) -> None:
    try:
        r = get_redis_pool()
        r.setex(_published_defense_redis_key(call_id), 172800, str(level))
    except Exception as e:
        logger.debug(f"记录已下发防御等级失败 call_id={call_id}: {e}")


def _defense_control_config(target_level: int, ui_message: str) -> dict:
    """与 upgrade_level 分支一致，用于升级与降级（含 target_level=1）。"""
    if target_level <= 1:
        return {
            "video_fps": 5.0,
            "ui_message": ui_message,
            "warning_mode": "inline",
            "block_call": False,
        }
    if target_level == 2:
        return {
            "video_fps": 15.0,
            "ui_message": ui_message,
            "warning_mode": "modal",
            "block_call": False,
        }
    return {
        "video_fps": 30.0,
        "ui_message": ui_message,
        "warning_mode": "fullscreen",
        "block_call": True,
    }


async def _finalize_previous_mdp_event(
    db,
    call_id: int,
    next_state: MDPState,
):
    """用当前状态回填上一条尚未完成的 MDP 决策事件。"""
    result = await db.execute(
        select(MDPDecisionEvent)
        .where(
            MDPDecisionEvent.call_id == call_id,
            MDPDecisionEvent.label_status == "open",
        )
        .order_by(MDPDecisionEvent.step_index.desc(), MDPDecisionEvent.event_id.desc())
        .limit(1)
    )
    previous_event = result.scalar_one_or_none()
    if not previous_event:
        return

    previous_state = MDPState.from_dict(json.loads(previous_event.state_json))
    action = DefenseAction(previous_event.action_level)
    previous_event.next_state_key = next_state.policy_key()
    previous_event.next_state_json = _json_dumps(next_state.to_dict())
    previous_event.reward = reward_builder.calculate_step_reward(
        current_state=previous_state,
        action=action,
        next_state=next_state,
    )
    previous_event.reward_source = "transition"
    previous_event.label_status = "resolved"


async def _finalize_open_mdp_events(
    db,
    call_id: int,
    final_result: DetectionResult | str,
):
    """在通话结束时为尚未闭合的 MDP 决策事件补充终局奖励。"""
    result = await db.execute(
        select(MDPDecisionEvent).where(
            MDPDecisionEvent.call_id == call_id,
            MDPDecisionEvent.label_status.in_(["open", "resolved"]),
        )
    )
    events = result.scalars().all()
    for event in events:
        action = DefenseAction(event.action_level)
        final_reward = reward_builder.calculate_final_outcome_reward(action, final_result)
        event.reward = round((event.reward or 0.0) + final_reward, 2)
        event.reward_source = "final"
        event.label_status = "finalized"


async def _create_mdp_decision_event(
    db,
    *,
    call_id: int,
    user_id: int,
    step_index: int,
    trigger_type: str,
    decision,
    observation: MDPObservation,
    llm_result: dict,
    rule_hit: dict,
    defense_payload: dict,
    ai_detection_log_id: int | None,
    message_log_id: int | None,
):
    event = MDPDecisionEvent(
        call_id=call_id,
        user_id=user_id,
        step_index=step_index,
        trigger_type=trigger_type,
        state_key=decision.state.policy_key(),
        state_json=_json_dumps(decision.state.to_dict()),
        fused_score=observation.risk_score,
        text_conf=observation.text_conf,
        audio_conf=observation.audio_conf,
        video_conf=observation.video_conf,
        llm_result_json=_json_dumps(llm_result),
        rule_hit_json=_json_dumps(rule_hit),
        action_level=decision.action.value,
        defense_payload_json=_json_dumps(defense_payload),
        policy_version=decision.policy_version,
        reason_codes_json=_json_dumps(decision.reason_codes),
        fallback_used=decision.fallback_used,
        ai_detection_log_id=ai_detection_log_id,
        message_log_id=message_log_id,
        label_status="open",
    )
    db.add(event)
    return event

def publish_realtime_score(user_id: int, call_id: int, detection_type: str, is_risk: bool, confidence: float):
    """向前端 WebSocket 实时推送综合检测分数
    
    从 Redis 读取各模态最新置信度，组合成前端期望的 detection_result 格式推送。
    """
    try:
        r = get_redis_pool()
        audio_conf = float(r.get(f"call:{call_id}:latest_audio_conf") or 0)
        video_conf = float(r.get(f"call:{call_id}:latest_video_conf") or 0)
        text_conf = float(r.get(f"call:{call_id}:latest_text_conf") or 0)

        if detection_type == "audio":
            audio_conf = confidence
        elif detection_type == "video":
            video_conf = confidence
        elif detection_type == "text":
            text_conf = confidence

        overall = max(audio_conf, video_conf, text_conf) * 100

        # 产品策略：音频/视频高风险仅用于“提高检测等级”，不直接弹窗打扰用户；
        # 只有文本/LLM 融合阶段给出高风险结论后，前端再展示风险弹窗。
        if detection_type in ("audio", "video"):
            is_fraud = False
        else:
            is_fraud = is_risk or (overall >= 60)

        payload = {
            "type": "detection_result",
            "data": {
                "overall_score": round(overall, 1),
                "voice_confidence": round(audio_conf, 4),
                "video_confidence": round(video_conf, 4),
                "text_confidence": round(text_conf, 4),
                "is_fraud": is_fraud,
                "advice": "" if not is_fraud else "检测到风险，请提高警惕",
                "keywords": [],
            }
        }
        message_data = {"user_id": user_id, "payload": payload}
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
        logger.info(f"Realtime score pushed: user={user_id}, type={detection_type}, audio={audio_conf:.2f}, video={video_conf:.2f}, text={text_conf:.2f}")
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

async def ensure_call_record_exists(db, call_id: int, user_id: int, auto_commit: bool = True) -> Optional[CallRecord]:
    """
    确保 CallRecord 存在，并返回通话记录对象
    
    Args:
        db: 数据库会话
        call_id: 通话ID
        user_id: 用户ID
        auto_commit: 是否自动提交事务，默认True。
                     在使用统一事务管理器时应设为False
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
        if auto_commit:
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

def apply_video_debounce(user_id: int, call_id: int, raw_is_fake: bool, confidence: float = 0.0) -> dict:
    """
    使用 Redis 实现置信度加权滑动窗口防抖和状态机
    
    优化点：
    - 存储置信度而非简单0/1
    - 加权投票：高置信度样本权重更大
    - 滞后区设计：避免状态抖动
    """
    try:
        r = get_redis_pool()
        # 存储置信度（用于加权投票）
        conf_window_key = f"detect:window_conf:{call_id}"
        # 保留二进制窗口用于兼容（可选）
        window_key = f"detect:window:{call_id}"
        state_key = f"detect:state:{call_id}"
        
        # 存储置信度值
        r.lpush(conf_window_key, confidence)
        r.ltrim(conf_window_key, 0, 4)  # 保留最近5帧
        r.expire(conf_window_key, 3600)
        
        # 兼容：仍存储二进制值
        val = 1 if raw_is_fake else 0
        r.lpush(window_key, val)
        r.ltrim(window_key, 0, 4)
        r.expire(window_key, 3600)
        
        # 读取置信度历史
        conf_history = r.lrange(conf_window_key, 0, -1)
        conf_values = [float(x) for x in conf_history]
        
        # 置信度加权投票
        # 策略：置信度>0.5的帧参与加权，权重=置信度
        weighted_sum = 0.0
        valid_count = 0
        for conf in conf_values:
            if conf > 0.5:  # 只统计疑似伪造的帧
                weighted_sum += conf
                valid_count += 1
        
        # 计算加权平均风险度
        avg_risk = weighted_sum / len(conf_values) if conf_values else 0.0
        
        # 读取历史状态
        prev_state_raw = r.get(state_key)
        prev_state = prev_state_raw if prev_state_raw else "SAFE"
        
        # 状态机转换（带滞后区）
        current_state = prev_state
        
        # ALARM阈值：加权平均>=0.6 且 至少3帧有检测
        if avg_risk >= 0.60 and valid_count >= 3:
            current_state = "ALARM"
        # SAFE阈值：加权平均<=0.3 或 有效帧<2
        elif avg_risk <= 0.30 or valid_count <= 1:
            current_state = "SAFE"
        # 0.3~0.6 为滞后区：保持原状态（避免抖动）
        else:
            current_state = prev_state
            logger.debug(f"视频防抖滞后区: avg_risk={avg_risk:.2f}, 保持状态={prev_state}")
        
        if current_state != prev_state:
            r.setex(state_key, 3600, current_state)
            logger.info(f"视频状态切换: {prev_state} -> {current_state} (avg_risk={avg_risk:.2f}, weighted_sum={weighted_sum:.2f})")
        
        return {
            "final_is_fake": (current_state == "ALARM"),
            "fake_count": valid_count,
            "avg_risk": round(avg_risk, 3),
            "weighted_sum": round(weighted_sum, 3),
            "state": current_state,
            "prev_state": prev_state
        }
    except Exception as e:
        logger.error(f"Debounce logic failed: {e}")
        return {"final_is_fake": raw_is_fake, "state": "UNKNOWN", "prev_state": "UNKNOWN", "fake_count": -1, "avg_risk": 0.0}


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
                    model_version=result.get('model_version', 'unknown')
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
                    _set_published_defense_level(call_id, 2)
                    
                    # 记录到Redis，供文本检测融合时参考
                    r = get_redis_pool()
                    r.setex(f"call:{call_id}:audio_high_risk_flag", 300, "1")
                    
                    logger.warning(f"音频检测到高危特征，已提升防御等级，等待文本融合判断 | 置信度: {confidence:.2f}")
                
                publish_realtime_score(user_id, call_id, "audio", is_fake, confidence)
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

                debounce_data = apply_video_debounce(user_id, call_id, raw_is_fake, raw_conf)
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
                        model_version=raw_result.get("model_version", "unknown")
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
                    _set_published_defense_level(call_id, 2)
                    
                    # 记录到Redis，供文本检测融合时参考
                    r.setex(f"call:{call_id}:video_high_risk_flag", 300, "1")
                    
                    logger.warning(f"视频检测到高危特征，已提升防御等级，等待文本融合判断 | 置信度: {raw_conf:.2f}")
                    
                publish_realtime_score(user_id, call_id, "video", curr_state == "ALARM", raw_conf)
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
    """文本触发器 & 大模型多模态融合决策司令部
    
    事务管理优化：
    - 所有数据库操作（AIDetectionLog、ChatMessage、CallRecord）在同一事务中
    - 外部服务调用（Redis发布、邮件发送）在事务commit后执行
    - 外部服务失败仅记录日志，不影响数据库数据一致性
    """
    bind_context(user_id=user_id, call_id=call_id)
    logger.info(f"Task started: MDP Fusion Command Center (Len: {len(text)})")

    async def _process():
        # 使用事务上下文管理器，支持 commit 后回调
        tx_ctx = TransactionContext()
        
        async with tx_ctx.begin() as db:
            try:
                record = await ensure_call_record_exists(db, call_id, user_id, auto_commit=False)
                
                if len(text.strip()) < 2:
                    return {"status": "skipped", "reason": "text too short"}
                    
                await memory_service.add_message(call_id, text)

                # 每次收到前端文本都持久化到 chat_messages，避免被后续分支提前 return 跳过
                seq_result = await db.execute(
                    select(ChatMessage.sequence)
                    .where(ChatMessage.call_id == call_id)
                    .order_by(ChatMessage.sequence.desc(), ChatMessage.message_id.desc())
                    .limit(1)
                )
                last_seq = seq_result.scalar_one_or_none() or 0
                current_sequence = last_seq + 1
                msg = ChatMessage(
                    call_id=call_id,
                    sequence=current_sequence,
                    speaker="other",
                    content=text[:1000] if len(text) > 1000 else text
                )
                db.add(msg)
                await db.flush()

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
                        uid, cid = user_id, call_id
                        kw = rule_hit["keyword"]

                        def _post_rule_upgrade():
                            publish_control_command(
                                uid,
                                {
                                    "type": "control",
                                    "action": "upgrade_level",
                                    "target_level": 3,
                                    "config": {
                                        "ui_message": f"触发高危防骗规则: {kw}",
                                        "block_call": True,
                                    },
                                },
                            )
                            _set_published_defense_level(cid, 3)

                        tx_ctx.add_post_commit_task(_post_rule_upgrade)
                        force_llm = True

                try:
                    text_result = await model_service.predict_text(text)
                    text_conf = text_result.get("confidence", 0.5) 
                    
                    if force_llm:
                        text_conf = max(text_conf, 0.95) 
                    elif text_conf < 0.3:
                        # 低风险快速返回前，同步更新实时文本置信度，避免前端长期显示 0%
                        r_fast = get_redis_pool()
                        r_fast.setex(f"call:{call_id}:latest_text_conf", 3600, str(text_conf))

                        # A 方案：即使 ONNX 判定低风险，也记录一次模型检测日志
                        detected_text = text[:500] if len(text) > 500 else text
                        sensitive_keywords = await _extract_sensitive_keywords(text)
                        ai_log = AIDetectionLog(
                            call_id=call_id,
                            detection_type="text",
                            text_confidence=text_conf,
                            overall_score=round(text_conf * 100.0, 2),
                            detected_text=detected_text,
                            detected_keywords=sensitive_keywords,
                            match_script=None,
                            intent="onnx_safe_skip",
                            time_offset=time_offset,
                            model_version=f"onnx-{text_result.get('model_type', 'unknown')}"
                        )
                        db.add(ai_log)
                        await db.flush()
                        # 推送一次实时融合分数（由 publish_realtime_score 读取 Redis 黑板值）
                        publish_realtime_score(user_id, call_id, "text", False, text_conf)
                        logger.info("ONNX 判定为安全闲聊，已记录检测日志并跳过重量级决策")
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
                
                # 获取环境信息
                env_info = environment_fusion_engine.get_environment_info(str(call_id))
                env_type = env_info.get("environment_type", "unknown")
                
                # 根据环境构建通话类型描述
                env_descriptions = {
                    "text_chat": "QQ/微信文字聊天场景",
                    "voice_chat": "QQ/微信语音聊天场景",
                    "phone_call": "电话通话场景",
                    "video_call": "视频通话场景",
                    "unknown": "视频通话场景" if is_video_call else "纯语音通话场景"
                }
                call_type_desc = env_descriptions.get(env_type, env_descriptions["unknown"])
                
                logger.info(f"[环境感知] 使用环境类型: {env_type}, 描述: {call_type_desc}")

                # 提取交互轮次
                chat_history = await memory_service.get_context(call_id)
                msg_count = len(chat_history.split('\n')) if isinstance(chat_history, str) else 1

                # 3. LLM 感知层 (只负责输出分类和意图)
                # 根据环境获取聊天双方信息（如果是文字聊天）
                chat_speakers = None
                if env_type == "text_chat":
                    # 尝试从Redis获取聊天双方信息
                    try:
                        r = get_redis_pool()
                        speakers_data = r.get(f"call:{call_id}:chat_speakers")
                        if speakers_data:
                            chat_speakers = json.loads(speakers_data)
                    except Exception:
                        pass
                
                llm_result = await llm_service.analyze_multimodal_risk(
                    user_input=text, 
                    chat_history=chat_history,
                    user_profile=user_profile_str,
                    call_type=call_type_desc,
                    audio_conf=audio_conf,
                    video_conf=f"{raw_video_conf:.4f}" if is_video_call else "N/A",
                    text_conf=text_conf,
                    chat_speakers=chat_speakers
                )
                
                # 4. 融合算分层 (使用环境感知融合引擎)
                fused_score = environment_fusion_engine.calculate_score(
                    llm_classification=llm_result,
                    local_text_conf=text_conf,
                    audio_conf=audio_conf,
                    video_conf=raw_video_conf,
                    call_id=str(call_id)  # 传入call_id用于环境感知和时序平滑
                )

                # 存储文本置信度供其他模态读取
                r.setex(f"call:{call_id}:latest_text_conf", 3600, str(text_conf))

                # 5. 马尔可夫决策层
                # 根据画像、计算的科学分数、交互轮次获取 O(1) 查表动作
                mdp_observation = MDPObservation(
                    user_id=user_id,
                    call_id=call_id,
                    risk_score=fused_score,
                    message_count=msg_count,
                    environment_type=env_type,
                    audio_risk_flag=audio_high_risk,
                    video_risk_flag=video_high_risk,
                    rule_hit_level=rule_hit.get("risk_level", 0) if rule_hit else 0,
                    text_conf=text_conf,
                    audio_conf=audio_conf,
                    video_conf=raw_video_conf,
                    llm_result=llm_result,
                    rule_hit=rule_hit or {},
                )
                decision = mdp_agent.select_action(user, mdp_observation)
                action_level = decision.action.value
                await _finalize_previous_mdp_event(db, call_id, decision.state)

                # 将日志数据落盘
                # 提取敏感关键词（从检测文本中）
                detected_text = text[:500] if len(text) > 500 else text  # 保存检测到的文本
                sensitive_keywords = await _extract_sensitive_keywords(text)
                
                ai_log = AIDetectionLog(
                    call_id=call_id,
                    detection_type="text",
                    text_confidence=fused_score / 100.0, # 兼容老数据库格式 
                    overall_score=fused_score,           # 得出的分数
                    detected_text=detected_text,         # 检测到的完整文本
                    detected_keywords=sensitive_keywords, # 敏感关键词
                    match_script=llm_result.get('match_script', None),  # 匹配的剧本
                    intent=llm_result.get('intent', None),              # 识别的意图
                    time_offset=time_offset,
                    model_version=f"fusion-mdp-v1-{text_result.get('model_type', 'unknown')}"
                )
                db.add(ai_log)
                await db.flush()

                # 6. 转化与执行 MDP 的拦截指令
                alert_level = 'safe'
                record_verdict = DetectionResult.SAFE
                ui_message = "请注意保护个人隐私。"
                defense_payload = {
                    "target_level": action_level + 1,
                    "ui_message": ui_message,
                    "warning_mode": "inline",
                    "block_call": False,
                }
                critical_message_log = None
                decision_message_log = None
                
                target_level = action_level + 1 # Action(0,1,2) 对应 System Level(1,2,3)

                if action_level == 2:
                    # MDP 裁定：三级最高防御 (强制阻断 + 联动监护人)
                    alert_level = 'critical'
                    record_verdict = DetectionResult.FAKE
                    ui_message = f"【极度高危】系统已阻断。防骗建议：{llm_result.get('advice', '请立刻停止操作！')}"
                    
                    # 触发联动闭环 - 发送 WebSocket 和邮件通知给监护人
                    # 使用事务内的 session 写入 MessageLog
                    critical_message_log = await notification_service.handle_detection_result(
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
                
                # 通知前端展示并在记录中更新（MessageLog 写入在事务内）
                decision_message_log = await notification_service.handle_detection_result(
                    db=db, user_id=user_id, call_id=call_id,
                    detection_type="MDP动态决策", is_risk=(action_level > 0), 
                    confidence=fused_score / 100.0,
                    risk_level=alert_level,
                    details=f"命中剧本: {llm_result.get('match_script', '无')} | 融合分: {fused_score:.1f}"
                )

                if record:
                    record.analysis = llm_result.get('analysis', '')
                    record.advice = llm_result.get('advice', '')
                    # 保存LLM识别的诈骗类型
                    fraud_type = llm_result.get('fraud_type', '')
                    if fraud_type and fraud_type != '其他':
                        record.fraud_type = fraud_type
                    
                    current_verdict = record.detected_result
                    if record_verdict == DetectionResult.FAKE:
                        record.detected_result = DetectionResult.FAKE
                    elif record_verdict == DetectionResult.SUSPICIOUS and current_verdict == DetectionResult.SAFE:
                        record.detected_result = DetectionResult.SUSPICIOUS
                    # 不再单独 commit，统一在事务结束时 commit

                # 将 WebSocket 控制指令注册为 commit 后执行（升级与自动降级：仅在实际等级变化时下发）
                prev_published = _get_published_defense_level(call_id)
                if target_level != prev_published:
                    cfg = _defense_control_config(target_level, ui_message)
                    payload_control = {
                        "type": "control",
                        "action": "upgrade_level",
                        "target_level": target_level,
                        "config": cfg,
                    }
                    defense_payload = payload_control

                    def _publish_and_remember(uid: int, pl: dict, cid: int, lvl: int):
                        publish_control_command(uid, pl)
                        _set_published_defense_level(cid, lvl)

                    tx_ctx.add_post_commit_task(_publish_and_remember, user_id, payload_control, call_id, target_level)

                # 注册 commit 后推送综合检测结果给前端
                _uid, _cid, _tc, _ac, _vc, _fs, _af = (
                    user_id, call_id, text_conf, audio_conf, raw_video_conf, fused_score, action_level
                )
                _kw = sensitive_keywords.split(",") if sensitive_keywords else []
                # 仅在融合分数达到高危，或 MDP 判定为最高动作时，前端触发高风险提示
                _is_high_risk_for_ui = (_fs >= 80.0) or (_af >= 2)
                _adv = llm_result.get('advice', '') if _is_high_risk_for_ui else ''

                def _push_detection_result():
                    det_payload = {
                        "type": "detection_result",
                        "data": {
                            "overall_score": round(_fs, 1),
                            "voice_confidence": round(_ac, 4),
                            "video_confidence": round(_vc, 4),
                            "text_confidence": round(_tc, 4),
                            "is_fraud": _is_high_risk_for_ui,
                            "advice": _adv,
                            "keywords": _kw,
                        }
                    }
                    msg = {"user_id": _uid, "payload": det_payload}
                    notification_service.redis.publish("fraud_alerts", json.dumps(msg, ensure_ascii=False))

                tx_ctx.add_post_commit_task(_push_detection_result)

                await db.flush()
                primary_message_log = decision_message_log or critical_message_log
                await _create_mdp_decision_event(
                    db,
                    call_id=call_id,
                    user_id=user_id,
                    step_index=msg_count,
                    trigger_type="text",
                    decision=decision,
                    observation=mdp_observation,
                    llm_result=llm_result,
                    rule_hit=rule_hit or {},
                    defense_payload=defense_payload,
                    ai_detection_log_id=ai_log.log_id,
                    message_log_id=primary_message_log.id if primary_message_log else None,
                )

                # 保存返回结果供后续使用
                result_data = {
                    "status": "success",
                    "fused_score": fused_score,
                    "action_level": action_level,
                    "policy_version": decision.policy_version,
                    "reason_codes": decision.reason_codes,
                    "fallback_used": decision.fallback_used,
                    "state_key": decision.state.policy_key(),
                }
                
            except Exception as e:
                logger.error(f"MDP Fusion Command Center failed: {e}", exc_info=True)
                return {"status": "error", "message": str(e), "call_id": call_id}
        
        # 事务已成功 commit，执行外部服务调用（Redis 发布、WebSocket 推送等）
        # 这些操作失败不会回滚数据库
        await tx_ctx.execute_post_commit_tasks()
        
        return result_data

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
                
                # 提取通话环境信息并更新call_record
                environment = ocr_result.get("environment", {})
                if environment:
                    platform_str = environment.get("platform", "other")
                    caller_number = environment.get("caller_number", "")
                    target_name = environment.get("target_name", "")
                    
                    # 转换平台字符串为枚举
                    platform_map = {
                        "wechat": CallPlatform.WECHAT,
                        "qq": CallPlatform.QQ,
                        "phone": CallPlatform.PHONE,
                        "video_call": CallPlatform.VIDEO_CALL,
                        "other": CallPlatform.OTHER
                    }
                    detected_platform = platform_map.get(platform_str.lower(), CallPlatform.OTHER)
                    
                    # 更新call_record（仅当字段为空时）
                    updated = False
                    if record.platform == CallPlatform.PHONE or record.platform is None:
                        record.platform = detected_platform
                        updated = True
                    if not record.caller_number and caller_number:
                        record.caller_number = caller_number
                        updated = True
                    if not record.target_name and target_name:
                        record.target_name = target_name
                        updated = True
                    
                    if updated:
                        await db.commit()
                        logger.info(f"更新通话环境: platform={detected_platform.value}, caller={caller_number or target_name}")
                    
                    # ===== 环境感知：设置融合引擎环境并推送给前端 =====
                    try:
                        # 判断是否为纯文字聊天（从OCR结果分析）
                        is_text_chat = False
                        if ocr_result.get("dialogues"):
                            # 如果有对话结构，检查是否为纯文字聊天界面
                            dialogues = ocr_result.get("dialogues", [])
                            # 文字聊天特征：有明确的发送者标识，消息气泡样式
                            has_chat_structure = len(dialogues) > 0 and any(
                                "sender" in d and "content" in d for d in dialogues
                            )
                            # 语音聊天特征：有通话时长、麦克风图标等
                            has_voice_indicators = any(
                                indicator in extracted_text for indicator in 
                                ["通话时长", "麦克风", "扬声器", "挂断", "免提"]
                            )
                            is_text_chat = has_chat_structure and not has_voice_indicators
                        
                        # 设置环境到融合引擎
                        environment_fusion_engine.set_call_environment(
                            call_id=str(call_id),
                            platform=platform_str,
                            is_text_chat=is_text_chat
                        )
                        
                        # 获取环境信息
                        env_info = environment_fusion_engine.get_environment_info(str(call_id))
                        
                        # 通过WebSocket推送给前端
                        payload = {
                            "type": "environment_detected",
                            "data": {
                                "call_id": call_id,
                                "platform": platform_str,
                                "is_text_chat": is_text_chat,
                                "environment_type": env_info["environment_type"],
                                "description": env_info["description"],
                                "active_modalities": env_info["active_modalities"],
                                "weights": env_info["weights"]
                            }
                        }
                        
                        message_data = {
                            "user_id": user_id,
                            "payload": payload
                        }
                        
                        r = get_redis_pool()
                        r.publish("fraud_alerts", json.dumps(message_data, ensure_ascii=False))
                        logger.info(f"[环境感知] 环境识别结果已推送到前端: call_id={call_id}, env={env_info['environment_type']}, is_text_chat={is_text_chat}")
                        
                        # 根据环境调整后续检测策略
                        if is_text_chat:
                            logger.info(f"[环境感知] 检测到文字聊天场景，建议前端关闭音频/视频检测，专注文本分析")
                        elif platform_str in ["wechat", "qq"]:
                            logger.info(f"[环境感知] 检测到{platform_str}语音聊天，启用音频+文本检测")
                        elif platform_str == "phone":
                            logger.info(f"[环境感知] 检测到电话通话，启用音频+文本检测")
                        elif platform_str == "video_call":
                            logger.info(f"[环境感知] 检测到视频通话，启用三模态检测")
                            
                    except Exception as env_e:
                        logger.warning(f"[环境感知] 环境设置或推送失败: {env_e}")
                
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


async def _get_chat_history_from_db(db, call_id: int) -> str:
    """当 Redis 短期记忆不可用时，从持久化消息表回退读取对话历史。"""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.call_id == call_id)
        .order_by(ChatMessage.sequence.asc(), ChatMessage.message_id.asc())
    )
    messages = result.scalars().all()
    if not messages:
        return ""

    history_lines = []
    for index, msg in enumerate(messages, start=1):
        speaker = (msg.speaker or "unknown").strip()
        content = (msg.content or "").strip()
        if not content:
            continue
        history_lines.append(f"{index}. [{speaker}] {content}")

    return "\n".join(history_lines)


@celery_app.task(name="generate_post_call_summary")
def generate_post_call_summary_task(call_id: int, user_id: int):
    """通话结束后，由 Celery 去读取自身内存中的对话并呼叫 LLM 进行复盘"""
    async def _process():
        try:
            async with AsyncSessionLocal() as db:
                chat_history = await memory_service.get_context(call_id)

                # Redis 上下文为空或读取失败时，回退到数据库中的持久化对话。
                if not chat_history or chat_history in ["暂无历史上下文。", "获取上下文失败。"]:
                    chat_history = await _get_chat_history_from_db(db, call_id)
                    if chat_history:
                        logger.info(f"通话 {call_id} 使用数据库对话历史生成总结")

                if not chat_history:
                    await memory_service.clear_context(call_id)
                    return {"status": "skipped", "reason": "No conversation history"}

                summary_result = await llm_service.generate_final_summary(chat_history)
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
                    await _finalize_open_mdp_events(db, call_id, record.detected_result)
                    await db.commit()
                    
                    # 提取并保存长期记忆
                    if record.detected_result in [DetectionResult.FAKE, DetectionResult.SUSPICIOUS]:
                        try:
                            detection_result = {
                                "is_fraud": record.detected_result == DetectionResult.FAKE,
                                "risk_level": final_risk,
                                "fraud_type": getattr(record, 'fraud_type', '未知类型'),
                                "match_script": getattr(record, 'match_script', ''),
                                "intent": ""
                            }
                            await long_term_memory_service.extract_and_save_memories(
                                db=db,
                                user_id=user_id,
                                call_id=call_id,
                                detection_result=detection_result
                            )
                            logger.info(f"已保存用户 {user_id} 的长期记忆")
                        except Exception as mem_e:
                            logger.error(f"保存长期记忆失败: {mem_e}")
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            await memory_service.clear_context(call_id)

    try:
        return async_to_sync(_process)()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# 辅助函数
# ============================================================================

SENSITIVE_KEYWORDS = [
    "转账", "汇款", "验证码", "密码", "银行卡", "信用卡", "账户",
    "屏幕共享", "远程控制", "下载", "链接", "点击", "APP", "安装",
    "安全账户", "冻结", "涉嫌", "违法", "法院", "公安", "警察", "逮捕"
]

async def _extract_sensitive_keywords(text: str) -> str:
    """从文本中提取敏感关键词"""
    found = []
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in text:
            found.append(keyword)
    return ",".join(found) if found else ""

async def _save_chat_message(db, call_id: int, sequence: int, speaker: str, content: str):
    """保存对话消息到数据库"""
    try:
        msg = ChatMessage(
            call_id=call_id,
            sequence=sequence,
            speaker=speaker,
            content=content[:1000] if len(content) > 1000 else content  # 限制长度
        )
        db.add(msg)
        await db.commit()
    except Exception as e:
        logger.error(f"保存对话消息失败: {e}")
        await db.rollback()