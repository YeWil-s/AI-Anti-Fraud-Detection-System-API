"""
用户模型
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from app.db.database import Base
import enum


class AdminRole(str, enum.Enum):
    """管理员角色枚举"""
    NONE = "none"           # 普通成员
    SECONDARY = "secondary" # 副管理员（可接收告警，有限权限）
    PRIMARY = "primary"     # 主管理员（完全权限）


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    phone = Column(String(20), unique=True, index=True, nullable=False, comment="手机号")
    email = Column(String(100), unique=True, index=True, nullable=True, comment="邮箱")
    username = Column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    name = Column(String(50), nullable=True, comment="用户姓名")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    family_id = Column(Integer, ForeignKey("family_groups.id"), nullable=True, index=True, comment="所属家庭组ID")
    role_type = Column(String(20), default="青壮年", index=True, comment="角色类型(如老人、儿童、学生、青壮年)")
    gender = Column(String(10), nullable=True, comment="性别(男/女/未知)")
    profession = Column(String(50), nullable=True, comment="职业")
    marital_status = Column(String(20), nullable=True, comment="婚姻状况(单身/已婚/离异)")
    is_active = Column(Boolean, default=True, comment="账号是否激活")
    is_admin = Column(Boolean, default=False, comment="是否为家庭管理员(监护人)")  # 保留兼容
    admin_role = Column(String(20), nullable=True, index=True, comment="管理员角色: none/secondary/primary")
    guardian_phone = Column(String(20), nullable=True, comment="监护人手机号")  # 备用监护人
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")
    
    @property
    def is_primary_admin(self) -> bool:
        """是否为主管理员"""
        return self.admin_role == AdminRole.PRIMARY.value
    
    @property
    def is_secondary_admin(self) -> bool:
        """是否为副管理员"""
        return self.admin_role == AdminRole.SECONDARY.value
    
    @property
    def is_any_admin(self) -> bool:
        """是否为任意管理员"""
        return self.admin_role in (AdminRole.PRIMARY.value, AdminRole.SECONDARY.value)
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, username={self.username}, role={self.role_type}, phone={self.phone})>"
