"""
通话记录模型
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from app.db.database import Base
import enum

class CallPlatform(str, enum.Enum):
    PHONE = "phone"      # 传统电话
    WECHAT = "wechat"    # 微信
    QQ = "qq"            # QQ
    OTHER = "other"

class DetectionResult(str, enum.Enum):
    """检测结果枚举"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    FAKE = "fake"


class CallRecord(Base):
    """通话记录表"""
    __tablename__ = "call_records"
    
    call_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True, comment="所属用户")
    caller_number = Column(String(20), nullable=True, comment="来电号码")
    platform = Column(SQLEnum(CallPlatform), default=CallPlatform.PHONE, comment="通话平台")
    target_name = Column(String(100), nullable=True, comment="对方昵称/备注")
    start_time = Column(DateTime(timezone=True), nullable=False, comment="通话开始时间")
    end_time = Column(DateTime(timezone=True), nullable=True, comment="通话结束时间")
    duration = Column(Integer, default=0, comment="通话时长(秒)")
    detected_result = Column(
        SQLEnum(DetectionResult),
        default=DetectionResult.SAFE,
        comment="检测结果(safe/suspicious/fake)"
    )
    audio_url = Column(String(500), nullable=True, comment="录音文件URL")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    
    def __repr__(self):
        return f"<CallRecord(call_id={self.call_id}, user_id={self.user_id}, result={self.detected_result})>"
