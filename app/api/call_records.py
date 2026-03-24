"""
通话记录管理API路由 - 数据隔离
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query,BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import Optional
from datetime import datetime  
from pydantic import BaseModel
from app.db.database import get_db, AsyncSessionLocal
from app.core.security import get_current_user_id
from app.models.message_log import MessageLog
from app.tasks.celery_app import celery_app
from app.models.call_record import CallRecord, DetectionResult, CallPlatform
from app.models.ai_detection_log import AIDetectionLog
from app.models.chat_message import ChatMessage
from app.models.user import User
from app.schemas import ResponseModel

from app.services.memory_service import memory_service
from app.services.llm_service import llm_service
from app.services.risk_fusion_engine import fusion_engine_v2, environment_fusion_engine
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/call-records", tags=["通话记录"])


@router.get("/my-records", response_model=ResponseModel)
async def get_my_call_records(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    result_filter: Optional[DetectionResult] = Query(None, description="按检测结果过滤"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取当前用户的通话记录"""
    query = select(CallRecord).where(CallRecord.user_id == current_user_id)
    
    if result_filter:
        query = query.where(CallRecord.detected_result == result_filter)
    
    query = query.order_by(CallRecord.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    records = result.scalars().all()
    
    # 统计总数
    count_result = await db.execute(
        select(func.count()).select_from(CallRecord).where(CallRecord.user_id == current_user_id)
    )
    total = count_result.scalar() or 0
    
    return ResponseModel(
        code=200,
        message="查询成功",
        data={
            "records": [
                {
                    "call_id": r.call_id,
                    "platform": r.platform.value if r.platform else "phone", # [新增] 返回平台
                    "target_name": r.target_name,       # [新增] 返回昵称
                    "caller_number": r.caller_number,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "end_time": r.end_time.isoformat() if r.end_time else None,
                    "duration": r.duration,
                    "detected_result": r.detected_result.value if r.detected_result else None,
                    "audio_url": r.audio_url,
                    "analysis": r.analysis, 
                    "advice": r.advice,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in records
            ],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size
            }
        }
    )

@router.post("/start", response_model=dict)
async def start_call(
    platform: str, 
    target_identifier: str, # 电话号或微信名
    user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """开始通话 (App端调用)"""
    
    # 简单的参数校验，防止枚举报错
    try:
        platform_enum = CallPlatform(platform)
    except ValueError:
        platform_enum = CallPlatform.OTHER

    new_call = CallRecord(
        user_id=user_id,
        platform=platform_enum,
        # 如果是电话，存 caller_number；如果是微信/QQ，存 target_name
        caller_number=target_identifier if platform_enum == CallPlatform.PHONE else None,
        target_name=target_identifier if platform_enum != CallPlatform.PHONE else None,
        start_time=datetime.now(),
        detected_result=DetectionResult.SAFE
    )
    db.add(new_call)
    await db.commit()
    await db.refresh(new_call)
    
    return {"call_id": new_call.call_id, "status": "started"}

@router.get("/record/{call_id}", response_model=ResponseModel)
async def get_call_record_detail(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取单个通话记录详情"""
    result = await db.execute(
        select(CallRecord).where(
            and_(
                CallRecord.call_id == call_id,
                CallRecord.user_id == current_user_id
            )
        )
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    
    # 查询AI检测日志
    log_result = await db.execute(
        select(AIDetectionLog).where(AIDetectionLog.call_id == call_id)
    )
    detection_log = log_result.scalar_one_or_none()
    
    return ResponseModel(
        code=200,
        message="查询成功",
        data={
            "call_record": {
                "call_id": record.call_id,
                "platform": record.platform.value if record.platform else "phone",
                "target_name": record.target_name,
                "caller_number": record.caller_number,
                "start_time": record.start_time.isoformat() if record.start_time else None,
                "end_time": record.end_time.isoformat() if record.end_time else None,
                "duration": record.duration,
                "detected_result": record.detected_result.value,
                "audio_url": record.audio_url,
                "analysis": record.analysis,  
                "advice": record.advice,
                "created_at": record.created_at.isoformat()
            },
            # 详情页通常需要更详细的AI数据
            "detection_log": {
                "overall_score": detection_log.overall_score,
                "voice_conf": detection_log.voice_confidence,
                "video_conf": detection_log.video_confidence,
                "keywords": detection_log.detected_keywords
            } if detection_log else None
        }
    )


@router.get("/family-records", response_model=ResponseModel)
async def get_family_call_records(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取家庭组成员的通话记录
    
    **数据隔离**: 只能查看同一家庭组成员的记录
    """
    # 查询当前用户
    user_result = await db.execute(
        select(User).where(User.user_id == current_user_id)
    )
    current_user = user_result.scalar_one_or_none()
    
    if not current_user or not current_user.family_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="您还未加入任何家庭组"
        )
    
    # 查询家庭组所有成员ID
    family_users_result = await db.execute(
        select(User.user_id).where(User.family_id == current_user.family_id)
    )
    family_user_ids = [row[0] for row in family_users_result.fetchall()]
    
    # 查询家庭组成员的通话记录
    query = select(CallRecord).where(CallRecord.user_id.in_(family_user_ids))
    query = query.order_by(CallRecord.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    records = result.scalars().all()
    
    # 统计总数
    count_result = await db.execute(
        select(func.count()).select_from(CallRecord).where(CallRecord.user_id.in_(family_user_ids))
    )
    total = count_result.scalar() or 0
    
    return ResponseModel(
        code=200,
        message="查询成功",
        data={
            "records": [
                {
                    "call_id": r.call_id,
                    "user_id": r.user_id,
                    "caller_number": r.caller_number,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "duration": r.duration,
                    "detected_result": r.detected_result.value if r.detected_result else None,
                    "analysis": r.analysis, 
                    "advice": r.advice,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                }
                for r in records
            ],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size
            }
        }
    )


@router.delete("/record/{call_id}", response_model=ResponseModel)
async def delete_call_record(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    删除通话记录
    
    **数据隔离**: 只能删除自己的记录
    """
    # 验证所有权
    result = await db.execute(
        select(CallRecord).where(
            and_(
                CallRecord.call_id == call_id,
                CallRecord.user_id == current_user_id
            )
        )
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="通话记录不存在或无权删除"
        )
    
    await db.delete(record)
    await db.commit()
    
    return ResponseModel(
        code=200,
        message="删除成功",
        data={"call_id": call_id}
    )

# 定义接收前端请求的数据结构
class CallRecordEndRequest(BaseModel):
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    cover_image: Optional[str] = None

@router.post("/{call_id}/end", response_model=ResponseModel)
async def end_call_record(
    call_id: int,
    payload: CallRecordEndRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    结束通话监测，更新最终音视频文件URL，并下发AI全局总结任务给Celery
    """
    # 1. 验证所有权
    result = await db.execute(
        select(CallRecord).where(
            and_(
                CallRecord.call_id == call_id,
                CallRecord.user_id == current_user_id
            )
        )
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="通话记录不存在或无权限"
        )
    
    # 2. 更新结束时间和通话时长
    record.end_time = datetime.now()
    if record.start_time:
        record.duration = int((record.end_time - record.start_time).total_seconds())
        
    # 3. 更新完整的音视频文件和封面 URL
    if payload.audio_url:
        record.audio_url = payload.audio_url
    if payload.video_url:
        record.video_url = payload.video_url
    if payload.cover_image:
        record.cover_image = payload.cover_image
        
    await db.commit()
    
    # ==========================================
    # [核心修改] 不再由 FastAPI 查内存，直接交给 Celery 决策
    # ==========================================
    celery_app.send_task("generate_post_call_summary", args=[call_id, current_user_id])
    
    # 持久化Redis中的对话历史到数据库
    await _persist_chat_history(db, call_id)
    
    # 清理风险融合引擎的时序缓存
    fusion_engine_v2.clear_temporal_cache(str(call_id))
    
    return ResponseModel(
        code=200, 
        message="通话记录归档成功，已交由异步引擎生成AI全局总结", 
        data={"duration": record.duration}
    )


async def _persist_chat_history(db: AsyncSession, call_id: int):
    """将Redis中的对话历史持久化到数据库"""
    try:
        # 从Redis获取对话历史
        chat_history = await memory_service.get_context(call_id)
        
        if not chat_history or chat_history == "暂无历史上下文。":
            logger.info(f"通话 {call_id} 没有对话历史需要持久化")
            return
        
        # 检查数据库中是否已有记录
        existing_result = await db.execute(
            select(ChatMessage).where(ChatMessage.call_id == call_id)
        )
        if existing_result.scalars().first():
            logger.info(f"通话 {call_id} 的对话历史已存在，跳过持久化")
            return
        
        # 解析并保存对话
        messages = []
        for line in chat_history.split('\n'):
            if line.strip() and '. ' in line:
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    seq_str, content = parts
                    try:
                        seq = int(seq_str)
                        msg = ChatMessage(
                            call_id=call_id,
                            sequence=seq,
                            speaker="other",  # Redis中不区分说话人
                            content=content[:1000] if len(content) > 1000 else content
                        )
                        messages.append(msg)
                    except ValueError:
                        continue
        
        if messages:
            for msg in messages:
                db.add(msg)
            await db.commit()
            logger.info(f"通话 {call_id} 的对话历史已持久化，共 {len(messages)} 条消息")
        
    except Exception as e:
        logger.error(f"持久化对话历史失败 call_id={call_id}: {e}")
        await db.rollback()

@router.get("/{call_id}/audit-logs", response_model=ResponseModel)
async def get_call_audit_logs(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """获取指定通话的详细审计日志（时间轴）"""
    # 1. 查询通话记录及其归属
    result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")

    # 2. 权限校验：本人或其家庭组管理员可看
    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()

    if record.user_id != current_user_id:
        owner_result = await db.execute(select(User).where(User.user_id == record.user_id))
        owner = owner_result.scalar_one_or_none()
        if not owner or owner.family_id != current_user.family_id or not current_user.is_admin:
            raise HTTPException(status_code=403, detail="无权查看该记录的审计日志")

    # 3. 获取底层 AI 检测日志
    ai_logs_result = await db.execute(
        select(AIDetectionLog).where(AIDetectionLog.call_id == call_id).order_by(AIDetectionLog.time_offset.asc())
    )
    ai_logs = ai_logs_result.scalars().all()

    # 4. 获取告警消息日志
    msg_logs_result = await db.execute(
        select(MessageLog).where(MessageLog.call_id == call_id).order_by(MessageLog.created_at.asc())
    )
    msg_logs = msg_logs_result.scalars().all()

    return ResponseModel(
        code=200,
        message="获取审计日志成功",
        data={
            "ai_events": [{
                "time_offset": log.time_offset,
                "overall_score": log.overall_score,
                "voice_conf": log.voice_confidence,
                "video_conf": log.video_confidence,
                "text_conf": log.text_confidence,
                "evidence_url": log.evidence_snapshot,
                "text_content": log.detected_keywords,
            } for log in ai_logs],
            "alert_events": [{
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "msg_type": log.msg_type,
                "risk_level": log.risk_level,
                "title": log.title,
                "content": log.content
            } for log in msg_logs]
        }
    )

@router.post("/{call_id}/report-to-admin", response_model=ResponseModel)
async def report_call_to_admin(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """家庭组长审查后，提交给系统管理员喂给智能体学习"""
    # 1. 权限校验
    result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")

    user_result = await db.execute(select(User).where(User.user_id == current_user_id))
    current_user = user_result.scalar_one_or_none()

    # 必须是管理员，且这条记录是自己家人的
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="仅家庭组管理员可提交审查")

    owner_result = await db.execute(select(User).where(User.user_id == record.user_id))
    owner = owner_result.scalar_one_or_none()
    if not owner or owner.family_id != current_user.family_id:
        raise HTTPException(status_code=403, detail="只能提交家庭组成员的记录")

    # 2. 将检测结果标记为 FAKE (欺诈)，以便 admin.py 中的 get_fraud_cases 接口能捕获到
    record.detected_result = DetectionResult.FAKE
    await db.commit()

    return ResponseModel(code=200, message="已成功提交至系统防诈特征库待审核队列")


@router.get("/{call_id}/chat-history", response_model=ResponseModel)
async def get_call_chat_history(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取通话的文本对话历史记录
    
    从数据库中获取持久化的对话记录，包含用户和对方的文本内容
    """
    # 1. 查询通话记录并校验权限
    result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")
    
    # 权限校验：本人或家庭组管理员
    if record.user_id != current_user_id:
        user_result = await db.execute(select(User).where(User.user_id == current_user_id))
        current_user = user_result.scalar_one_or_none()
        owner_result = await db.execute(select(User).where(User.user_id == record.user_id))
        owner = owner_result.scalar_one_or_none()
        if not owner or owner.family_id != current_user.family_id or not current_user.is_admin:
            raise HTTPException(status_code=403, detail="无权查看该记录的对话历史")
    
    # 2. 从数据库获取对话历史（优先），Redis作为实时补充
    try:
        # 先尝试从数据库获取持久化的对话
        db_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.call_id == call_id)
            .order_by(ChatMessage.sequence.asc())
        )
        db_messages = db_result.scalars().all()
        
        messages = []
        for msg in db_messages:
            messages.append({
                "sequence": msg.sequence,
                "speaker": msg.speaker,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
            })
        
        # 如果数据库没有，尝试从Redis获取（实时通话中）
        source = "database"
        if not messages:
            chat_history = await memory_service.get_context(call_id)
            if chat_history and chat_history != "暂无历史上下文。":
                source = "redis"
                for line in chat_history.split('\n'):
                    if line.strip() and '. ' in line:
                        parts = line.split('. ', 1)
                        if len(parts) == 2:
                            seq, content = parts
                            messages.append({
                                "sequence": int(seq),
                                "speaker": "other",  # Redis中不区分说话人
                                "content": content,
                                "timestamp": None
                            })
        
        return ResponseModel(
            code=200,
            message="获取对话历史成功",
            data={
                "call_id": call_id,
                "message_count": len(messages),
                "messages": messages,
                "source": source
            }
        )
    except Exception as e:
        logger.error(f"获取对话历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取对话历史失败")


@router.get("/{call_id}/detection-timeline", response_model=ResponseModel)
async def get_detection_timeline(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取详细的检测时间轴数据
    
    提供每个时间点的三模态详细检测数据，用于前端展示检测过程曲线
    """
    # 1. 查询通话记录并校验权限
    result = await db.execute(
        select(CallRecord).where(CallRecord.call_id == call_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")
    
    # 权限校验
    if record.user_id != current_user_id:
        user_result = await db.execute(select(User).where(User.user_id == current_user_id))
        current_user = user_result.scalar_one_or_none()
        owner_result = await db.execute(select(User).where(User.user_id == record.user_id))
        owner = owner_result.scalar_one_or_none()
        if not owner or owner.family_id != current_user.family_id or not current_user.is_admin:
            raise HTTPException(status_code=403, detail="无权查看该记录的检测时间轴")
    
    # 2. 获取AI检测日志（按时间排序）
    ai_logs_result = await db.execute(
        select(AIDetectionLog)
        .where(AIDetectionLog.call_id == call_id)
        .order_by(AIDetectionLog.time_offset.asc())
    )
    ai_logs = ai_logs_result.scalars().all()
    
    # 3. 构建时间轴数据
    timeline = []
    for log in ai_logs:
        timeline.append({
            "time_offset": log.time_offset,
            "timestamp": log.created_at.isoformat() if log.created_at else None,
            "modalities": {
                "text": {
                    "confidence": log.text_confidence,
                    "score": log.text_confidence * 100 if log.text_confidence else 0
                },
                "voice": {
                    "confidence": log.voice_confidence,
                    "score": log.voice_confidence * 100 if log.voice_confidence else 0
                },
                "video": {
                    "confidence": log.video_confidence,
                    "score": log.video_confidence * 100 if log.video_confidence else 0
                }
            },
            "overall_score": log.overall_score,
            "fused_risk_level": _get_risk_level(log.overall_score),
            "detected_text": log.detected_text,  # 检测到的完整文本
            "detected_keywords": log.detected_keywords,  # 敏感关键词
            "match_script": log.match_script,  # 匹配的剧本
            "intent": log.intent,  # 识别的意图
            "detection_type": log.detection_type,
            "model_version": log.model_version,
            "evidence_url": log.evidence_snapshot,
            "log_id": log.log_id
        })
    
    # 4. 计算统计数据
    stats = {
        "total_events": len(timeline),
        "max_overall_score": max([t["overall_score"] for t in timeline]) if timeline else 0,
        "avg_text_conf": sum([t["modalities"]["text"]["confidence"] for t in timeline]) / len(timeline) if timeline else 0,
        "avg_voice_conf": sum([t["modalities"]["voice"]["confidence"] for t in timeline]) / len(timeline) if timeline else 0,
        "avg_video_conf": sum([t["modalities"]["video"]["confidence"] for t in timeline]) / len(timeline) if timeline else 0,
        "duration_seconds": timeline[-1]["time_offset"] if timeline else 0
    }
    
    return ResponseModel(
        code=200,
        message="获取检测时间轴成功",
        data={
            "call_id": call_id,
            "timeline": timeline,
            "statistics": stats
        }
    )


def _get_risk_level(score: float) -> str:
    """根据分数获取风险等级"""
    if score >= 80:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"


@router.get("/{call_id}/evidence/{log_id}")
async def get_evidence_detail(
    call_id: int,
    log_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取指定检测日志的详细证据信息
    
    包括证据截图、检测详情、技术参数等
    """
    # 1. 查询检测日志
    result = await db.execute(
        select(AIDetectionLog).where(
            and_(
                AIDetectionLog.log_id == log_id,
                AIDetectionLog.call_id == call_id
            )
        )
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="检测日志不存在")
    
    # 2. 查询通话记录并校验权限
    record_result = await db.execute(
        select(CallRecord).where(CallRecord.call_id == call_id)
    )
    record = record_result.scalar_one_or_none()
    
    if record.user_id != current_user_id:
        user_result = await db.execute(select(User).where(User.user_id == current_user_id))
        current_user = user_result.scalar_one_or_none()
        owner_result = await db.execute(select(User).where(User.user_id == record.user_id))
        owner = owner_result.scalar_one_or_none()
        if not owner or owner.family_id != current_user.family_id or not current_user.is_admin:
            raise HTTPException(status_code=403, detail="无权查看该证据")
    
    # 3. 构建证据详情
    evidence_detail = {
        "log_id": log.log_id,
        "call_id": log.call_id,
        "time_offset": log.time_offset,
        "created_at": log.created_at.isoformat() if log.created_at else None,
        "detection_type": log.detection_type,
        "modalities": {
            "text": {
                "confidence": log.text_confidence,
                "detected_text": log.detected_text,  # 检测到的完整文本
                "detected_keywords": log.detected_keywords,  # 敏感关键词
                "match_script": log.match_script,  # 匹配的剧本
                "intent": log.intent  # 识别的意图
            },
            "voice": {
                "confidence": log.voice_confidence
            },
            "video": {
                "confidence": log.video_confidence
            }
        },
        "overall_score": log.overall_score,
        "risk_level": _get_risk_level(log.overall_score),
        "evidence": {
            "snapshot_url": log.evidence_snapshot,
            "ocr_text": log.image_ocr_text if log.detection_type == "image" else None,
            "ocr_dialogue_hash": log.ocr_dialogue_hash if log.detection_type == "image" else None
        },
        "technical_details": {
            "algorithm_details": log.algorithm_details,
            "model_version": log.model_version
        }
    }
    
    return ResponseModel(
        code=200,
        message="获取证据详情成功",
        data=evidence_detail
    )


@router.get("/{call_id}/environment", response_model=ResponseModel)
async def get_call_environment(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取通话环境信息（平台类型、启用的检测模态等）
    
    前端可根据此信息动态调整检测任务
    """
    # 1. 查询通话记录
    result = await db.execute(
        select(CallRecord).where(
            CallRecord.call_id == call_id,
            CallRecord.user_id == current_user_id
        )
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")
    
    # 2. 获取环境信息
    env_info = environment_fusion_engine.get_environment_info(str(call_id))
    
    # 3. 补充平台信息
    env_info["platform"] = record.platform.value if record.platform else "unknown"
    env_info["target_name"] = record.target_name
    env_info["caller_number"] = record.caller_number
    
    return ResponseModel(
        code=200,
        message="获取环境信息成功",
        data=env_info
    )


@router.post("/{call_id}/environment", response_model=ResponseModel)
async def set_call_environment(
    call_id: int,
    platform: str = Query(..., description="平台类型: wechat/qq/phone/video_call/other"),
    is_text_chat: bool = Query(False, description="是否为纯文字聊天"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    设置通话环境（通常由后端OCR识别后自动调用，也可由前端手动设置）
    """
    # 1. 查询通话记录
    result = await db.execute(
        select(CallRecord).where(
            CallRecord.call_id == call_id,
            CallRecord.user_id == current_user_id
        )
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")
    
    # 2. 设置环境
    environment_fusion_engine.set_call_environment(
        call_id=str(call_id),
        platform=platform,
        is_text_chat=is_text_chat
    )
    
    # 3. 获取更新后的环境信息
    env_info = environment_fusion_engine.get_environment_info(str(call_id))
    
    # 4. 通过WebSocket推送给前端
    try:
        from app.services.notification_service import notification_service
        import json
        
        payload = {
            "type": "environment_detected",
            "data": {
                "call_id": call_id,
                "platform": platform,
                "is_text_chat": is_text_chat,
                "environment_type": env_info["environment_type"],
                "description": env_info["description"],
                "active_modalities": env_info["active_modalities"]
            }
        }
        
        message_data = {
            "user_id": current_user_id,
            "payload": payload
        }
        
        notification_service.redis.publish("fraud_alerts", json.dumps(message_data))
        logger.info(f"环境识别结果已推送到前端: call_id={call_id}, env={env_info['environment_type']}")
    except Exception as e:
        logger.warning(f"WebSocket推送环境信息失败: {e}")
    
    return ResponseModel(
        code=200,
        message="环境设置成功",
        data=env_info
    )


class EmergencyAlertRequest(BaseModel):
    call_id: int
    alert_type: str = "emergency"  # emergency: 紧急报警, suspicious: 可疑行为
    message: Optional[str] = None


@router.post("/{call_id}/emergency-alert", response_model=ResponseModel)
async def trigger_emergency_alert(
    call_id: int,
    request: EmergencyAlertRequest,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    一键报警/紧急求助
    
    当用户遭遇高风险诈骗时，触发紧急报警：
    1. 向家庭组所有管理员发送紧急通知（WebSocket + 邮件）
    2. 记录报警日志
    3. 可选：触发自动录音保存证据
    """
    from app.models.family_group import FamilyAdmin, FamilyGroup
    from app.services.notification_service import notification_service
    from app.services.email_service import email_service
    import json
    
    # 1. 验证通话记录存在且属于当前用户
    result = await db.execute(
        select(CallRecord).where(
            and_(
                CallRecord.id == call_id,
                CallRecord.user_id == current_user_id
            )
        )
    )
    call_record = result.scalar_one_or_none()
    
    if not call_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="通话记录不存在"
        )
    
    # 2. 获取用户信息
    user_result = await db.execute(
        select(User).where(User.user_id == current_user_id)
    )
    user = user_result.scalar_one_or_none()
    
    if not user or not user.family_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户未加入家庭组，无法发送报警"
        )
    
    # 3. 查询家庭组所有管理员
    admins_result = await db.execute(
        select(FamilyAdmin, User)
        .join(User, FamilyAdmin.user_id == User.user_id)
        .where(FamilyAdmin.family_id == user.family_id)
    )
    admins = admins_result.all()
    
    if not admins:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="家庭组没有管理员，无法发送报警"
        )
    
    # 4. 构造紧急报警消息
    timestamp = datetime.now().isoformat()
    alert_message = request.message or f"用户 {user.name or user.username} 触发了一键报警，可能正在遭遇诈骗！"
    
    emergency_payload = {
        "type": "emergency_alert",
        "data": {
            "title": "🚨 紧急报警",
            "message": alert_message,
            "victim_id": current_user_id,
            "victim_name": user.name or user.username,
            "victim_phone": user.phone,
            "call_id": call_id,
            "family_id": user.family_id,
            "timestamp": timestamp,
            "display_mode": "fullscreen",  # 全屏显示
            "action": "alarm",  # 响铃 + 震动
            "alert_type": request.alert_type
        }
    }
    
    # 5. 向所有管理员发送紧急通知
    notified_count = 0
    for admin_record, admin_user in admins:
        # WebSocket 推送
        notification_service._publish_to_redis(admin_user.user_id, emergency_payload)
        notified_count += 1
        logger.info(f"紧急报警已发送给管理员 {admin_user.user_id} ({admin_record.admin_role})")
        
        # 邮件通知
        if admin_user.email:
            try:
                await email_service.send_guardian_alert(
                    to_email=admin_user.email,
                    victim_name=user.name or user.username,
                    risk_level="critical",
                    details=f"【一键报警】{alert_message}"
                )
            except Exception as e:
                logger.warning(f"发送报警邮件失败: {e}")
    
    # 6. 记录报警日志
    from app.models.message_log import MessageLog
    alert_log = MessageLog(
        user_id=current_user_id,
        call_id=call_id,
        msg_type="emergency_alert",
        risk_level="critical",
        title="一键报警",
        content=alert_message,
        is_read=False
    )
    db.add(alert_log)
    await db.commit()
    
    logger.info(f"一键报警处理完成: 通知了 {notified_count} 位管理员")
    
    return ResponseModel(
        code=200,
        message=f"紧急报警已发送，已通知 {notified_count} 位家庭管理员",
        data={
            "alert_id": alert_log.id,
            "notified_admins": notified_count,
            "timestamp": timestamp
        }
    )