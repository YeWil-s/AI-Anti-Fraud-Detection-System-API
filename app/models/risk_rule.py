"""
风险规则模型
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from app.db.database import Base


class RiskRule(Base):
    """风险规则表"""
    __tablename__ = "risk_rules"
    
    rule_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    keyword = Column(String(100), nullable=False, index=True, comment="高危话术关键词")
    action = Column(String(20), nullable=False, comment="提醒动作(alert/block)")
    risk_level = Column(Integer, default=1, comment="风险等级(1-5)")
    is_active = Column(Boolean, default=True, comment="规则是否启用")
    description = Column(String(255), nullable=True, comment="规则描述")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    def __repr__(self):
        return f"<RiskRule(rule_id={self.rule_id}, keyword={self.keyword}, action={self.action})>"
