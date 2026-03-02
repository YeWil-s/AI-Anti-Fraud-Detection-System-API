from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.database import Base

class FamilyGroup(Base):
    """家庭组表"""
    __tablename__ = "family_groups"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    group_name = Column(String(100), nullable=False, comment="家庭组名称")
    owner_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, comment="组长/创建者ID")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")

    def __repr__(self):
        return f"<FamilyGroup(id={self.id}, name={self.group_name}, owner={self.owner_id})>"