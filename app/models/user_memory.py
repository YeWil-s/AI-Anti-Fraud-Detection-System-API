"""
用户长期记忆模型
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum


class MemoryType(str, enum.Enum):
    """记忆类型枚举"""
    FRAUD_EXPERIENCE = "fraud_experience"  # 遭遇过的诈骗经历
    ALERT_RESPONSE = "alert_response"      # 对告警的反应
    PREFERENCE = "preference"              # 用户偏好
    RISK_PATTERN = "risk_pattern"          # 风险行为模式
    LEARNING_PROGRESS = "learning_progress" # 学习进度


class UserMemory(Base):
    """用户长期记忆表"""
    __tablename__ = "user_memories"
    
    memory_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    memory_type = Column(String(30), nullable=False, comment="记忆类型")
    content = Column(Text, nullable=False, comment="记忆内容")
    importance = Column(Integer, default=3, comment="重要性 1-5")
    source_call_id = Column(Integer, ForeignKey("call_records.call_id", ondelete="SET NULL"), nullable=True, comment="来源通话ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系
    user = relationship("User", back_populates="memories")
    
    def __repr__(self):
        return f"<UserMemory(id={self.memory_id}, user={self.user_id}, type={self.memory_type})>"
