"""
通知与报警服务
功能：分级处理检测结果、日志存库、监护人应用内实时联动
"""
import json
from datetime import datetime
from typing import Optional

import redis
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message_log import MessageLog
from app.models.user import User
from app.core.config import settings
from app.core.logger import get_logger
from app.core.defense_levels import (
    get_display_mode,
    should_notify_guardian,
    should_send_email,
    DEFENSE_LEVEL_MAP
)
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
        details: str = "",
        websocket_push: bool = True,
    ) -> Optional[MessageLog]:
        """
        处理检测结果：记录日志并触发实时预警
        
        【重要】MessageLog 与 AIDetectionLog 职责边界：
        
        - AIDetectionLog：
          * 由检测任务层 (detection_tasks.py) 直接写入
          * 存储原始检测数据：置信度分数、模型版本、检测文本、证据快照等
          * 用于审计、模型分析、报告生成
          * 本服务不应写入 AIDetectionLog
          
        - MessageLog：
          * 由本通知服务 (notification_service) 写入
          * 存储用户可见的通知消息：标题、内容、风险等级、已读状态
          * 用于前端消息列表展示、未读计数
          * 不应包含原始模型数据（避免数据冗余）
        
        【幂等性保证】
        在写入 MessageLog 前检查是否已存在相同 call_id + msg_type 的记录
        如果已存在则跳过写入，避免 Celery 任务重试时产生重复数据
        
        【事务管理】
        本方法不单独 commit，由调用方统一管理事务
        
        websocket_push:
            为 False 时不向当前用户发 WebSocket（避免与 detect_text 融合后的 alert 重复弹窗），
            仍写入 MessageLog，仍可按策略通知监护人/邮件。
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

        # 2. [幂等性检查] 避免 Celery 重试时重复写入
        existing = await db.execute(
            select(MessageLog).where(
                and_(
                    MessageLog.call_id == call_id,
                    MessageLog.msg_type == msg_type,
                    MessageLog.title == title  # 同一类型同一标题表示同一事件
                )
            )
        )
        existing_log = existing.scalar_one_or_none()
        if existing_log:
            logger.info(f"Notification already exists for call_id={call_id}, msg_type={msg_type}, title={title}, skipping duplicate")
            # 幂等性保证：已存在则跳过数据库写入，但仍进行 WebSocket 推送和监护人联动
            result_log = existing_log
        else:
            # 3. [存库] 记录至 MessageLog
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
            # 不单独 commit，由调用方统一管理事务
            logger.info(f"MessageLog added for User {user_id}: {title}")
            result_log = new_log

        # 4. [WebSocket] 发送给当前正在通话的用户
        # 使用统一的防御等级映射获取显示模式
        display_mode = get_display_mode(risk_level)
        
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
        if websocket_push:
            self._publish_to_redis(user_id, ws_payload)

        # 5. [监护人联动] 使用统一的防御等级判断是否通知家人
        if is_risk and should_notify_guardian(risk_level):
            await self._notify_family_in_app(db, user_id, ws_payload)
            
        # 6. [邮件通知] 使用统一的防御等级判断是否发送邮件
        if is_risk and should_send_email(risk_level):
            await self._notify_guardian_by_email(db, user_id, risk_level, details)

        return result_log
            
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
        # 使用统一的防御等级映射获取显示模式和动作
        risk_level = original_payload['data']['risk_level']
        display_mode = get_display_mode(risk_level)
        
        # 根据风险等级设置动作
        action_map = {
            'critical': 'alarm',   # 响铃 + 震动
            'high': 'vibrate',     # 震动
        }
        action = action_map.get(risk_level, 'none')
        
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
            # [离线兜底] 给每位管理员写入消息日志，避免管理员不在线时丢提醒
            existing = await db.execute(
                select(MessageLog).where(
                    and_(
                        MessageLog.user_id == admin_user.user_id,
                        MessageLog.call_id == original_payload["data"].get("call_id"),
                        MessageLog.msg_type == "family_alert",
                        MessageLog.title == guardian_payload["data"]["title"],
                    )
                )
            )
            if not existing.scalar_one_or_none():
                db.add(
                    MessageLog(
                        user_id=admin_user.user_id,
                        call_id=original_payload["data"].get("call_id"),
                        msg_type="family_alert",
                        risk_level=risk_level,
                        title=guardian_payload["data"]["title"],
                        content=guardian_payload["data"]["message"],
                        is_read=False,
                    )
                )
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
        
        已统一委托给 email_service.send_guardian_alert_by_family 处理
        """
        await email_service.send_guardian_alert_by_family(
            db=db,
            user_id=current_user_id,
            risk_level=risk_level,
            details=details
        )

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