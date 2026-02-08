"""
通知与报警服务
负责分级处理、日志记录、短信通知和WebSocket推送
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
import json
import redis

from app.models.message_log import MessageLog
from app.models.user import User
from app.core.sms import send_fraud_alert_sms
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class NotificationService:
    
    def __init__(self):
        self.redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def handle_detection_result(
        self, 
        db: AsyncSession, 
        user_id: int, 
        call_id: int, 
        detection_type: str, # audio/video/text
        is_risk: bool, 
        confidence: float,
        risk_level: str = "low",
        details: str = ""
    ):
        """
        处理检测结果：分级报警、存库、通知
        """
        # 1. 准备基础数据
        timestamp = datetime.now().isoformat()
        title = ""
        content = ""
        msg_type = "info"
        
        # 2. 确定消息内容和类型
        if is_risk:
            msg_type = "alert"
            title = f"检测到{detection_type}异常风险"
            content = f"系统检测到疑似伪造内容 (置信度: {confidence:.2f})。{details}"
        else:
            msg_type = "info"
            title = f"{detection_type}检测通过"
            content = f"当前通话环境安全 (置信度: {confidence:.2f})。"

        # 3. [存库] 记录到 MessageLog (所有级别的报警都记录，便于事后审计)
        new_log = MessageLog(
            user_id=user_id,
            call_id=call_id,
            msg_type=msg_type,
            risk_level=risk_level,
            title=title,
            content=content,
            is_read=False
        )
        db.add(new_log)
        try:
            await db.commit()
            logger.info(f"Message logged: {title} (User: {user_id})")
        except Exception as e:
            logger.error(f"Failed to save message log: {e}")
            await db.rollback()

        # 4. [WebSocket] 构造前端弹窗/提示 payload
        # 只有中高风险才让前端弹窗(popup)，低风险只显示toast或静默
        display_mode = "popup" if risk_level in ["critical", "high", "medium"] else "toast"
        
        ws_payload = {
            "type": msg_type, # alert / info
            "data": {
                "title": title,
                "message": content,
                "risk_level": risk_level,
                "confidence": confidence,
                "call_id": call_id,
                "timestamp": timestamp,
                "display_mode": display_mode # 指示前端如何展示
            }
        }
        self._publish_to_redis(user_id, ws_payload)

        # 5. [短信通知] 中高风险 (critical, high) -> 通知家庭组管理员
        # 题目要求: "检测到中高风险的通话，则立即发送短信消息给家庭组的管理员"
        if risk_level in ["critical", "high", "medium"] and is_risk:
            await self._notify_family_admin(db, user_id, risk_level)
            
    async def _notify_family_admin(self, db: AsyncSession, current_user_id: int, risk_level: str):
        """通知家庭组管理员(或所有其他成员)"""
        # 查询当前用户以获取 family_id
        result = await db.execute(select(User).where(User.user_id == current_user_id))
        user = result.scalar_one_or_none()
        
        if not user or not user.family_id:
            return

        # 查询同家庭组的其他成员 (假设这些是管理员/监护人)
        # 实际项目中可能有 is_admin 字段，这里简化为通知所有家人
        family_members = await db.execute(
            select(User).where(
                User.family_id == user.family_id,
                User.user_id != current_user_id # 排除自己
            )
        )
        members = family_members.scalars().all()
        
        for member in members:
            send_fraud_alert_sms(
                phone=member.phone,
                name=user.name or user.username,
                risk_level=risk_level,
                time_str=datetime.now().strftime("%H:%M")
            )

    def _publish_to_redis(self, user_id: int, payload: dict):
        """推送到 Redis 供 Main 进程转发 WebSocket"""
        try:
            message_data = {
                "user_id": user_id,
                "payload": payload
            }
            self.redis.publish("fraud_alerts", json.dumps(message_data))
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")

notification_service = NotificationService()