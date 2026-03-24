from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean, DateTime
from sqlalchemy.sql import func
from app.db.database import Base

class KnowledgeItem(Base):
    """
    防诈知识库（元数据表）
    用于在前端展示视频/图文列表，并供推荐系统筛选
    """
    __tablename__ = "knowledge_items"
    
    id = Column(Integer, primary_key=True, index=True)
    item_type = Column(String(50))           # 资源类型: 'video' (视频) 或 'article' (图文)
    title = Column(String(255), nullable=False) # 标题
    summary = Column(Text)                   # 简介
    content_url = Column(String(500))        # 视频链接或图文详情页链接
    fraud_type = Column(String(100))         # 诈骗类型标签 (与向量库保持一致，如：刷单返利诈骗)
    target_group = Column(String(255))       # 目标受众标签 (如：学生,宝妈)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserLearningRecord(Base):
    """
    用户学习记录表
    记录谁学了什么，是否学完
    """
    __tablename__ = "user_learning_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    item_id = Column(Integer, ForeignKey("knowledge_items.id", ondelete="CASCADE"), index=True)
    is_completed = Column(Boolean, default=False) # 是否完成学习
    learned_at = Column(DateTime(timezone=True), onupdate=func.now()) # 最近学习/完成时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())