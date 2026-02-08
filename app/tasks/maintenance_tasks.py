"""
系统维护定时任务 (Celery)
负责清理旧日志、过期记录等
"""
from app.tasks.celery_app import celery_app
from app.db.database import AsyncSessionLocal
from app.models.ai_detection_log import AIDetectionLog
from app.models.message_log import MessageLog
from sqlalchemy import delete
from datetime import datetime, timedelta
import asyncio
from app.core.logger import get_logger

logger = get_logger(__name__)

@celery_app.task(name="clean_old_logs")
def clean_old_logs_task(days_to_keep: int = 30):
    """
    清理超过指定天数的旧日志
    默认保留 30 天
    """
    logger.info(f"Starting database cleanup task (Keep days: {days_to_keep})")
    
    async def _process():
        async with AsyncSessionLocal() as db:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # 1. 清理 AI 检测流水日志 (量最大)
                result_ai = await db.execute(
                    delete(AIDetectionLog).where(AIDetectionLog.created_at < cutoff_date)
                )
                deleted_ai_count = result_ai.rowcount
                
                # 2. 清理消息通知日志
                result_msg = await db.execute(
                    delete(MessageLog).where(MessageLog.created_at < cutoff_date)
                )
                deleted_msg_count = result_msg.rowcount
                
                await db.commit()
                
                logger.info(f"Cleanup finished. Deleted: {deleted_ai_count} AI logs, {deleted_msg_count} messages.")
                return {"status": "success", "deleted_ai": deleted_ai_count, "deleted_msg": deleted_msg_count}
                
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}", exc_info=True)
                await db.rollback()
                return {"status": "error", "message": str(e)}

    # 在 Celery 同步环境中运行异步 DB 操作
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_process())
    finally:
        loop.close()