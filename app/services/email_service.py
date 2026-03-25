"""
邮件服务模块
功能：异步发送邮件通知，支持 HTML 格式
"""
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import aiosmtplib
from typing import Optional, Dict, Any

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class EmailService:
    """邮件服务类"""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_tls = settings.SMTP_TLS
        self.email_from = settings.EMAIL_FROM
        self.email_from_name = settings.EMAIL_FROM_NAME
    
    async def send_email(
        self, 
        to_email: str, 
        subject: str, 
        body: str, 
        html_body: Optional[str] = None
    ) -> bool:
        """
        发送邮件
        
        Args:
            to_email: 收件人邮箱
            subject: 邮件主题
            body: 纯文本内容
            html_body: HTML 内容（可选）
            
        Returns:
            bool: 发送是否成功
        """
        if not self.smtp_user or not self.smtp_password:
            logger.warning("邮件服务未配置，跳过发送")
            return False
        
        try:
            # 创建邮件消息
            message = MIMEMultipart("alternative")
            message["From"] = formataddr((self.email_from_name, self.email_from))
            message["To"] = to_email
            message["Subject"] = subject
            
            # 添加纯文本内容
            message.attach(MIMEText(body, "plain", "utf-8"))
            
            # 添加 HTML 内容（如果提供）
            if html_body:
                message.attach(MIMEText(html_body, "html", "utf-8"))
            
            # 连接到 SMTP 服务器并发送
            # 根据端口选择连接方式：465端口使用SSL，587端口使用STARTTLS
            # aiosmtplib 参数说明：
            #   use_tls=True: 直接SSL连接（465端口）
            #   start_tls=True: 升级到TLS连接（587端口）
            if self.smtp_port == 465:
                # 465端口使用直接SSL连接
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.smtp_user,
                    password=self.smtp_password,
                    use_tls=True,
                )
            else:
                # 587或其他端口使用STARTTLS
                await aiosmtplib.send(
                    message,
                    hostname=self.smtp_host,
                    port=self.smtp_port,
                    username=self.smtp_user,
                    password=self.smtp_password,
                    start_tls=self.smtp_tls,  # 根据配置决定是否启用STARTTLS
                )
            
            logger.info(f"邮件发送成功: {to_email}, 主题: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False
    
    async def send_guardian_alert(
        self,
        to_email: str,
        victim_name: str,
        risk_level: str,
        risk_type: str = "诈骗风险",
        details: str = ""
    ) -> bool:
        """
        发送监护人预警邮件
        
        Args:
            to_email: 监护人邮箱
            victim_name: 受害者姓名
            risk_level: 风险等级 (high/critical/medium)
            risk_type: 风险类型
            details: 详细信息
        """
        # 根据风险等级设置颜色
        color_map = {
            "critical": "#dc3545",  # 红色
            "high": "#fd7e14",      # 橙色
            "medium": "#ffc107",    # 黄色
        }
        color = color_map.get(risk_level.lower(), "#6c757d")
        
        # 风险等级中文映射
        level_map = {
            "critical": "极高",
            "high": "高",
            "medium": "中",
            "low": "低",
        }
        level_cn = level_map.get(risk_level.lower(), risk_level)
        
        subject = f"【紧急预警】您的家人遭遇{level_cn}风险"
        
        # 纯文本内容
        body = f"""
尊敬的家庭监护人：

您的家人【{victim_name}】当前遭遇{level_cn}{risk_type}！

风险详情：{details or '系统检测到异常通话/交互行为'}

请立即联系家人确认安全，必要时采取阻断措施。

---
此邮件由 AI 反诈系统自动发送
        """
        
        # HTML 内容
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .alert-box {{ 
            background-color: {color}; 
            color: white; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 20px;
        }}
        .alert-box h1 {{ margin: 0; font-size: 24px; }}
        .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .footer {{ margin-top: 20px; font-size: 12px; color: #6c757da; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="alert-box">
            <h1>🚨 家人安全预警</h1>
            <p>风险等级：<strong>{level_cn}</strong></p>
        </div>
        
        <div class="content">
            <p>尊敬的家庭监护人：</p>
            <p>您的家人 <strong>{victim_name}</strong> 当前遭遇 <span style="color: {color}; font-weight: bold;">{level_cn}{risk_type}</span>！</p>
            
            <h3>风险详情：</h3>
            <p>{details or '系统检测到异常通话/交互行为，可能存在诈骗风险。'}</p>
            
            <h3>建议措施：</h3>
            <ul>
                <li>立即电话联系家人确认安全</li>
                <li>提醒家人不要转账或泄露个人信息</li>
                <li>必要时挂断可疑通话</li>
                <li>如已受骗，立即报警处理</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>---</p>
            <p>此邮件由 AI 反诈系统自动发送</p>
            <p>发送时间：系统自动生成</p>
        </div>
    </div>
</body>
</html>
        """
        
        return await self.send_email(to_email, subject, body, html_body)

    async def send_guardian_alert_by_family(
        self,
        db: AsyncSession,
        user_id: int,
        risk_level: str,
        call_info: Optional[Dict[str, Any]] = None,
        details: str = ""
    ) -> int:
        """
        【统一邮件通知接口】向用户所属家庭组的所有管理员发送预警邮件
        
        完整流程：
        1. 查询用户信息及所属 family_id
        2. 查询 FamilyAdmin 表找到该家庭的所有管理员
        3. 查询每个管理员的邮箱地址
        4. 向有邮箱的管理员发送告警邮件
        
        Args:
            db: 数据库会话
            user_id: 当前用户ID（潜在受害者）
            risk_level: 风险等级 (critical/high/medium/low)
            call_info: 通话相关信息字典，可包含 call_id, caller_number 等
            details: 风险详情描述
            
        Returns:
            int: 成功发送邮件的数量
        """
        # 延迟导入避免循环依赖
        from app.models.user import User
        from app.models.family_group import FamilyAdmin
        
        try:
            # 1. 查询受害者信息
            result = await db.execute(select(User).where(User.user_id == user_id))
            victim = result.scalar_one_or_none()
            
            if not victim:
                logger.info(f"用户 {user_id} 不存在，跳过邮件通知")
                return 0
                
            if not victim.family_id:
                logger.info(f"用户 {user_id} 未加入家庭组，跳过邮件通知")
                return 0

            # 2. 查询该家庭组有邮箱的管理员（排除受害者本人）
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
                logger.info(f"用户 {user_id} 的家庭组没有配置邮箱的管理员")
                return 0
            
            # 3. 向每个管理员发送邮件
            victim_name = victim.name or victim.username
            success_count = 0
            
            for admin_record, admin_user in admins:
                if admin_user.email:
                    sent = await self.send_guardian_alert(
                        to_email=admin_user.email,
                        victim_name=victim_name,
                        risk_level=risk_level,
                        details=details
                    )
                    if sent:
                        success_count += 1
                        logger.info(
                            f"已向管理员 {admin_user.user_id} (role={admin_record.admin_role}) "
                            f"发送邮件预警，受害者: {user_id}"
                        )
                        
            return success_count
            
        except Exception as e:
            logger.error(f"发送家庭组邮件预警失败: {e}")
            return 0


# 全局单例
email_service = EmailService()
