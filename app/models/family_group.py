from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLEnum, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum

class ApplicationStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class FamilyGroup(Base):
    """家庭组表"""
    __tablename__ = "family_groups"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    group_name = Column(String(100), nullable=False, comment="家庭组名称")
    admin_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, comment="主管理员ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    
    # 关系
    admins = relationship("FamilyAdmin", back_populates="family", cascade="all, delete-orphan")


class FamilyAdmin(Base):
    """家庭组管理员表（记录用户在哪些家庭组是管理员）"""
    __tablename__ = "family_admins"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    family_id = Column(Integer, ForeignKey("family_groups.id", ondelete="CASCADE"), nullable=False)
    admin_role = Column(String(20), nullable=False, comment="管理员角色: primary/secondary")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="添加时间")
    
    __table_args__ = (
        UniqueConstraint('user_id', 'family_id', name='uq_family_admin'),
    )
    
    # 关系
    user = relationship("User", back_populates="admin_families")
    family = relationship("FamilyGroup", back_populates="admins")
    
    @property
    def is_primary(self) -> bool:
        return self.admin_role == "primary"
    
    @property
    def is_secondary(self) -> bool:
        return self.admin_role == "secondary"


class FamilyApplication(Base):
    """家庭组加入申请表"""
    __tablename__ = "family_applications"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    family_id = Column(Integer, ForeignKey("family_groups.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)     
    status = Column(SQLEnum(ApplicationStatus), default=ApplicationStatus.PENDING)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<FamilyApplication(id={self.id}, family_id={self.family_id}, user_id={self.user_id}, status={self.status})>"
