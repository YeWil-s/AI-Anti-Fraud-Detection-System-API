"""
对话消息模型 - 存储通话过程中的文本对话历史
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.db.database import Base


class ChatMessage(Base):
    """对话消息表"""
    __tablename__ = "chat_messages"
    
    message_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("call_records.call_id", ondelete="CASCADE"), 
                     nullable=False, index=True, comment="对应通话ID")
    sequence = Column(Integer, nullable=False, comment="消息序号")
    speaker = Column(String(20), nullable=False, comment="说话人: user/other/system")
    content = Column(Text, nullable=False, comment="消息内容")
    # 消息实际发送/发生时间（由客户端或任务系统提供）
    timestamp = Column(DateTime(timezone=True), nullable=True, 
                       comment="消息实际发送/发生时间（由客户端或任务系统提供）")
    # 数据库入库时间（系统自动生成）
    created_at = Column(DateTime(timezone=True), server_default=func.now(),
                        comment="数据库入库时间（系统自动生成）")
    
    def __repr__(self):
        return f"<ChatMessage(msg_id={self.message_id}, call_id={self.call_id}, seq={self.sequence})>"
