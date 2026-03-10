"""
通话记录管理API路由 - 数据隔离
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query,BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import Optional
from datetime import datetime  
from pydantic import BaseModel
from app.db.database import get_db
from app.core.security import get_current_user_id
from app.models.message_log import MessageLog

from app.models.call_record import CallRecord, DetectionResult, CallPlatform
from app.models.ai_detection_log import AIDetectionLog
from app.models.user import User
from app.schemas import ResponseModel

from app.services.memory_service import memory_service
from app.services.llm_service import llm_service

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

async def generate_call_summary_background(call_id: int, chat_history: list):
    """在后台执行的大模型全量总结任务"""
    if not chat_history or len(chat_history) == 0:
        memory_service.clear_context(call_id)
        return
    
    try:
        # 1. 呼叫大模型进行全盘总结 (需要在 llm_service 中实现此方法)
        summary_result = await llm_service.generate_final_summary(chat_history)
        
        # 2. 开启独立的数据库会话进行更新
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
            record = result.scalar_one_or_none()
            
            if record:
                # 覆盖实时的 analysis 和 advice，填入全局总结版
                record.analysis = summary_result.get("analysis", record.analysis)
                record.advice = summary_result.get("advice", record.advice)
                
                # 如果大模型根据全局判断发现了深层套路，还可以做最后一次风险定性修正
                final_risk = summary_result.get("risk_level", "safe")
                current_verdict = record.detected_result
                
                if final_risk in ['fake', 'high', 'critical']:
                    record.detected_result = DetectionResult.FAKE
                elif final_risk in ['suspicious', 'medium'] and current_verdict == DetectionResult.SAFE:
                    record.detected_result = DetectionResult.SUSPICIOUS
                    
                await db.commit()
    except Exception as e:
        print(f"后台生成通话全局总结失败: {e}")
    finally:
        # 3. 无论成功失败，必须清理内存中的对话上下文，防止内存泄漏
        memory_service.clear_context(call_id)

# 定义接收前端请求的数据结构
class CallRecordEndRequest(BaseModel):
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    cover_image: Optional[str] = None

@router.post("/{call_id}/end", response_model=ResponseModel)
async def end_call_record(
    call_id: int,
    payload: CallRecordEndRequest,
    background_tasks: BackgroundTasks,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    结束通话监测，更新最终音视频文件URL，并按需触发AI全局总结
    """
    # 1. 验证所有权，确保只能操作自己的记录
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
    # [核心修改] 判断是否需要触发后台总结
    # ==========================================
    # 检查数据库中是否已经有大模型写好的评价和建议 (非空判断)
    has_llm_evaluation = bool(record.analysis and record.advice)
    
    chat_history = memory_service.get_context(call_id)
    
    if chat_history and not has_llm_evaluation:
        # 只有在“有对话记录” 且 “大模型从未评价过” 的情况下，才触发最终总结
        background_tasks.add_task(generate_call_summary_background, call_id, chat_history)
        msg = "通话记录归档成功，正在后台生成AI全局总结"
    else:
        # 如果没有说话，或者实时检测过程中大模型已经写过评价了，直接清空 Redis 记忆并跳过总结
        memory_service.clear_context(call_id)
        msg = "通话记录归档成功"
    
    # 4. 立即返回，不阻塞前端
    return ResponseModel(
        code=200, 
        message=msg, 
        data={"duration": record.duration}
    )
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
                "evidence_url": log.evidence_snapshot
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