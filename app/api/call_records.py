"""
通话记录管理API路由 - 数据隔离
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import Optional

from app.db.database import get_db
from app.core.security import get_current_user_id
from app.models.call_record import CallRecord, DetectionResult
from app.models.ai_detection_log import AIDetectionLog
from app.models.user import User
from app.schemas import ResponseModel

router = APIRouter(prefix="/api/call-records", tags=["通话记录"])


@router.get("/my-records", response_model=ResponseModel)
async def get_my_call_records(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    result_filter: Optional[DetectionResult] = Query(None, description="按检测结果过滤"),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取当前用户的通话记录
    
    **数据隔离**: 只返回当前登录用户的通话记录
    """
    # 核心:强制按user_id过滤
    query = select(CallRecord).where(CallRecord.user_id == current_user_id)
    
    if result_filter:
        query = query.where(CallRecord.detected_result == result_filter)
    
    # 排序和分页
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
                    "caller_number": r.caller_number,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "end_time": r.end_time.isoformat() if r.end_time else None,
                    "duration": r.duration,
                    "detected_result": r.detected_result.value if r.detected_result else None,
                    "audio_url": r.audio_url,
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


@router.get("/record/{call_id}", response_model=ResponseModel)
async def get_call_record_detail(
    call_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db)
):
    """
    获取单个通话记录详情
    
    **数据隔离**: 验证通话记录所有权,防止越权访问
    """
    # 核心:同时检查call_id和user_id
    result = await db.execute(
        select(CallRecord).where(
            and_(
                CallRecord.call_id == call_id,
                CallRecord.user_id == current_user_id  # 确保只能访问自己的记录
            )
        )
    )
    record = result.scalar_one_or_none()
    
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="通话记录不存在或无权访问"  # 不泄露具体原因
        )
    
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
                "caller_number": record.caller_number,
                "start_time": record.start_time.isoformat() if record.start_time else None,
                "end_time": record.end_time.isoformat() if record.end_time else None,
                "duration": record.duration,
                "detected_result": record.detected_result.value if record.detected_result else None,
                "audio_url": record.audio_url,
                "created_at": record.created_at.isoformat() if record.created_at else None
            },
            "detection_log": {
                "log_id": detection_log.log_id,
                "voice_confidence": detection_log.voice_confidence,
                "video_confidence": detection_log.video_confidence,
                "text_confidence": detection_log.text_confidence,
                "overall_score": detection_log.overall_score,
                "detected_keywords": detection_log.detected_keywords,
                "model_version": detection_log.model_version,
                "created_at": detection_log.created_at.isoformat() if detection_log.created_at else None
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
