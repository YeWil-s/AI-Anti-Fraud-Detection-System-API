"""
用户模型
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.db.database import Base


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    phone = Column(String(20), unique=True, index=True, nullable=False, comment="手机号")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    name = Column(String(50), nullable=True, comment="用户姓名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    family_id = Column(Integer, nullable=True, index=True, comment="家庭组ID")
    is_active = Column(Boolean, default=True, comment="账号是否激活")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, username={self.username}, phone={self.phone})>"
