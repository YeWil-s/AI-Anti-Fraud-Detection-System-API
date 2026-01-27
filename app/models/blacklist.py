"""
号码黑名单模型
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.db.database import Base


class NumberBlacklist(Base):
    """号码黑名单表"""
    __tablename__ = "number_blacklist"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    number = Column(String(20), unique=True, index=True, nullable=False, comment="电话号码")
    source = Column(String(50), nullable=False, comment="来源(official/user_report)")
    report_count = Column(Integer, default=1, comment="被举报次数")
    risk_level = Column(Integer, default=1, comment="风险等级(1-5)")
    is_active = Column(Boolean, default=True, comment="是否启用")
    description = Column(String(255), nullable=True, comment="备注说明")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    def __repr__(self):
        return f"<NumberBlacklist(id={self.id}, number={self.number}, source={self.source})>"
