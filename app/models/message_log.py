"""
个人消息/报警日志模型
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.sql import func
from app.db.database import Base

class MessageLog(Base):
    """个人消息日志表"""
    __tablename__ = "message_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True, comment="接收用户")
    call_id = Column(Integer, nullable=True, index=True, comment="关联通话ID")
    
    # 消息类型: alert(警告), info(提示), system(系统)
    msg_type = Column(String(20), nullable=False, default="info", comment="消息类型")
    
    # 风险等级: critical(高危), high(高), medium(中), low(低), safe(安全)
    risk_level = Column(String(20), default="safe", comment="风险等级")
    
    title = Column(String(100), nullable=False, comment="消息标题")
    content = Column(Text, nullable=False, comment="消息内容")
    is_read = Column(Boolean, default=False, comment="是否已读")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    
    def __repr__(self):
        return f"<MessageLog(id={self.id}, user={self.user_id}, type={self.msg_type})>"