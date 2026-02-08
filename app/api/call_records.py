"""
通话记录管理API路由 - 数据隔离
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import Optional
from datetime import datetime  

from app.db.database import get_db
from app.core.security import get_current_user_id

from app.models.call_record import CallRecord, DetectionResult, CallPlatform
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
