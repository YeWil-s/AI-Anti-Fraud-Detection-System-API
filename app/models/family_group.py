from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from app.db.database import Base
import enum

class ApplicationStatus(str, enum.Enum):
    PENDING = "pending"     # 待审批
    APPROVED = "approved"   # 已同意
    REJECTED = "rejected"   # 已拒绝

class FamilyGroup(Base):
    """家庭组表"""
    __tablename__ = "family_groups"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    group_name = Column(String(100), nullable=False, comment="家庭组名称")
    admin_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, comment="组长/创建者ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")

class FamilyApplication(Base):
    """家庭组加入申请表"""
    __tablename__ = "family_applications"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    family_id = Column(Integer, ForeignKey("family_groups.id"), nullable=False) # 关联你的家庭组
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)     
    status = Column(SQLEnum(ApplicationStatus), default=ApplicationStatus.PENDING)
    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<FamilyApplication(id={self.id}, family_id={self.family_id}, user_id={self.user_id}, status={self.status})>"