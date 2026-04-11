"""
消息中心 API 路由
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user_id
from app.db.database import get_db
from app.models.message_log import MessageLog
from app.schemas import ResponseModel
from app.core.time_utils import isoformat_bj

router = APIRouter(prefix="/api/messages", tags=["消息中心"])


@router.get("/my", response_model=ResponseModel)
async def get_my_messages(
    unread_only: bool = Query(False, description="是否只看未读"),
    since_id: int | None = Query(None, ge=1, description="增量拉取起始ID(不含)"),
    limit: int = Query(20, ge=1, le=100),
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    query = select(MessageLog).where(MessageLog.user_id == current_user_id)
    if unread_only:
        query = query.where(MessageLog.is_read == False)
    if since_id is not None:
        query = query.where(MessageLog.id > since_id)

    query = query.order_by(MessageLog.id.desc()).limit(limit)
    result = await db.execute(query)
    items = result.scalars().all()

    data = [
        {
            "id": m.id,
            "call_id": m.call_id,
            "msg_type": m.msg_type,
            "risk_level": m.risk_level,
            "title": m.title,
            "content": m.content,
            "is_read": m.is_read,
            "created_at": isoformat_bj(m.created_at),
        }
        for m in items
    ]
    return ResponseModel(code=200, message="查询成功", data={"items": data})


@router.get("/my/unread-count", response_model=ResponseModel)
async def get_my_unread_count(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(func.count(MessageLog.id)).where(
            MessageLog.user_id == current_user_id,
            MessageLog.is_read == False,
        )
    )
    count = result.scalar() or 0
    return ResponseModel(code=200, message="查询成功", data={"unread_count": count})


@router.post("/{message_id}/read", response_model=ResponseModel)
async def mark_message_read(
    message_id: int,
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(MessageLog).where(
            MessageLog.id == message_id,
            MessageLog.user_id == current_user_id,
        )
    )
    message = result.scalar_one_or_none()
    if not message:
        raise HTTPException(status_code=404, detail="消息不存在")

    if not message.is_read:
        message.is_read = True
        await db.commit()

    return ResponseModel(code=200, message="已标记为已读", data={"id": message_id})


@router.post("/read-all", response_model=ResponseModel)
async def mark_all_messages_read(
    current_user_id: int = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        update(MessageLog)
        .where(
            MessageLog.user_id == current_user_id,
            MessageLog.is_read == False,
        )
        .values(is_read=True)
    )
    await db.commit()
    affected = result.rowcount or 0
    return ResponseModel(code=200, message="全部标记为已读", data={"updated": affected})
