"""
通知与报警服务
功能：分级处理检测结果、日志存库、监护人应用内实时联动
"""
import json
from datetime import datetime
import redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message_log import MessageLog
from app.models.user import User
from app.core.config import settings
from app.core.logger import get_logger
from app.services.email_service import email_service

logger = get_logger(__name__)

class NotificationService:
    
    def __init__(self):
        # 使用同步 Redis 客户端进行消息发布，性能极高
        self.redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def handle_detection_result(
        self, 
        db: AsyncSession, 
        user_id: int, 
        call_id: int, 
        detection_type: str, 
        is_risk: bool, 
        confidence: float,
        risk_level: str = "low",
        details: str = ""
    ):
        """
        处理检测结果：记录日志并触发实时预警
        """
        timestamp = datetime.now().isoformat()
        
        # 1. 确定消息属性
        if is_risk:
            msg_type = "alert"
            title = f"检测到{detection_type}异常风险"
            content = f"当前通话环境存在伪造风险，未通过{detection_type}安全检测。{details}"
        else:
            msg_type = "info"
            title = f"{detection_type}检测通过"
            content = f"当前通话环境安全，未检测到异常{detection_type}特征。"

        # 2. [存库] 记录至 MessageLog (受控于任务层的边缘触发逻辑，不会产生冗余写入)
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
            logger.info(f"Log saved for User {user_id}: {title}")
        except Exception as e:
            logger.error(f"Failed to save message log: {e}")
            await db.rollback()

        # 3. [WebSocket] 发送给当前正在通话的用户
        # 中高风险显示弹窗 (popup)，低风险仅显示通知 (toast)
        display_mode = "popup" if risk_level in ["critical", "high", "medium"] else "toast"
        
        ws_payload = {
            "type": msg_type,
            "data": {
                "title": title,
                "message": content,
                "risk_level": risk_level,
                "confidence": confidence,
                "call_id": call_id,
                "timestamp": timestamp,
                "display_mode": display_mode
            }
        }
        self._publish_to_redis(user_id, ws_payload)

        # 4. [监护人联动] 如果检测到高风险(Level 3)或中高风险(Level 2)且确定有风险，发送应用内消息给家庭成员
        # Level 1 (medium) 不通知家人，只在用户端提示
        if is_risk and risk_level in ["critical", "high"]:
            await self._notify_family_in_app(db, user_id, ws_payload)
            
        # 5. [邮件通知] 高风险(Level 3)时发送邮件给监护人
        if is_risk and risk_level in ["critical"]:
            await self._notify_guardian_by_email(db, user_id, risk_level, details)
            
    async def _notify_family_in_app(self, db: AsyncSession, current_user_id: int, original_payload: dict):
        """
        核心任务：通过 WebSocket 向家庭组管理员发送预警
        
        智能通知逻辑：
        1. 查询被检测用户的 family_id
        2. 查询该家庭组的所有管理员（family_admins表）
        3. 通知所有管理员（主/副管理员都通知）
        """
        from app.models.family_group import FamilyAdmin, FamilyGroup
        
        # 1. 查询受害者信息
        result = await db.execute(select(User).where(User.user_id == current_user_id))
        victim = result.scalar_one_or_none()
        
        if not victim or not victim.family_id:
            return

        # 2. 查询该家庭组的所有管理员
        admins_result = await db.execute(
            select(FamilyAdmin, User)
            .join(User, FamilyAdmin.user_id == User.user_id)
            .where(FamilyAdmin.family_id == victim.family_id)
        )
        admins = admins_result.all()
        
        if not admins:
            logger.warning(f"User {current_user_id} 的家庭组没有管理员")
            return
        
        # 3. 构造给监护人的特定预警 Payload
        # 根据风险等级决定显示模式
        risk_level = original_payload['data']['risk_level']
        if risk_level == 'critical':
            display_mode = 'fullscreen'  # Level 3: 全屏
            action = 'alarm'  # 响铃 + 震动
        elif risk_level == 'high':
            display_mode = 'popup'  # Level 2: 弹窗
            action = 'vibrate'  # 震动
        else:
            display_mode = 'toast'  # Level 1: 提示
            action = 'none'
        
        guardian_payload = {
            "type": "family_alert",
            "data": {
                "title": "🚨 家人安全预警" if risk_level == 'critical' else "⚠️ 家人安全预警",
                "message": f"您的家人【{victim.name or victim.username}】疑似正在遭遇诈骗。{original_payload['data']['message']}",
                "risk_level": risk_level,
                "victim_id": current_user_id,
                "victim_name": victim.name or victim.username,
                "victim_phone": victim.phone,
                "call_id": original_payload['data'].get('call_id'),
                "family_id": victim.family_id,
                "timestamp": original_payload['data']['timestamp'],
                "display_mode": display_mode,
                "action": action  # 前端可执行的动作：none/vibrate/alarm
            }
        }
        
        # 4. 逐一发布消息到 Redis 频道
        for admin_record, admin_user in admins:
            self._publish_to_redis(admin_user.user_id, guardian_payload)
            logger.info(f"Sent family alert for User {current_user_id} to Admin {admin_user.user_id} (role={admin_record.admin_role})")

    async def _notify_guardian_by_email(
        self, 
        db: AsyncSession, 
        current_user_id: int, 
        risk_level: str,
        details: str = ""
    ):
        """
        通过邮件向监护人发送预警
        """
        from app.models.family_group import FamilyAdmin
        
        try:
            # 1. 查询受害者信息
            result = await db.execute(select(User).where(User.user_id == current_user_id))
            victim = result.scalar_one_or_none()
            
            if not victim or not victim.family_id:
                return

            # 2. 查询该家庭组有邮箱的管理员
            admins_result = await db.execute(
                select(FamilyAdmin, User)
                .join(User, FamilyAdmin.user_id == User.user_id)
                .where(
                    and_(
                        FamilyAdmin.family_id == victim.family_id,
                        User.email.isnot(None)
                    )
                )
            )
            admins = admins_result.all()
            
            if not admins:
                logger.info(f"User {current_user_id} 的家庭组没有配置邮箱的管理员")
                return
            
            # 3. 向每个管理员发送邮件
            victim_name = victim.name or victim.username
            
            for admin_record, admin_user in admins:
                if admin_user.email:
                    await email_service.send_guardian_alert(
                        to_email=admin_user.email,
                        victim_name=victim_name,
                        risk_level=risk_level,
                        details=details
                    )
                    logger.info(f"Sent email alert for User {current_user_id} to Admin {admin_user.user_id} (role={admin_record.admin_role})")
                    
        except Exception as e:
            logger.error(f"Failed to send guardian email: {e}")

    def _publish_to_redis(self, user_id: int, payload: dict):
        """
        底层传输：将消息发布到 Redis，由 FastAPI 主进程通过 WebSocket 转发
        """
        try:
            message_data = {
                "user_id": user_id,
                "payload": payload
            }
            # 发布到 Redis 的特定频道，确保 Main 进程能监听并推送到对应用户的 WebSocket
            self.redis.publish("fraud_alerts", json.dumps(message_data))
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")

# 全局单例
notification_service = NotificationService()