"""
用户模型
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.database import Base


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    phone = Column(String(20), unique=True, index=True, nullable=False, comment="手机号")
    email = Column(String(100), unique=True, index=True, nullable=True, comment="邮箱")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    name = Column(String(50), nullable=True, comment="用户姓名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    family_id = Column(Integer, ForeignKey("family_groups.id"), nullable=True, index=True, comment="所属家庭组ID（普通成员只能有一个）")
    role_type = Column(String(20), default="青壮年", index=True, comment="角色类型(如老人、儿童、学生、青壮年)")
    gender = Column(String(10), nullable=True, comment="性别(男/女/未知)")
    profession = Column(String(50), nullable=True, comment="职业")
    marital_status = Column(String(20), nullable=True, comment="婚姻状况(单身/已婚/离异)")
    is_active = Column(Boolean, default=True, comment="账号是否激活")
    is_admin = Column(Boolean, default=False, comment="是否为某家庭组的管理员")
    memory_summary = Column(Text, nullable=True, comment="长期记忆摘要（由系统自动生成）")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    # 关系
    admin_families = relationship("FamilyAdmin", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan", order_by="UserMemory.importance.desc()")
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, username={self.username}, role={self.role_type}, phone={self.phone})>"
