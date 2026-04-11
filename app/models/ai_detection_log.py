"""
AI检测日志模型
"""
from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, Text, String
from sqlalchemy.sql import func
from app.db.database import Base


class AIDetectionLog(Base):
    """AI检测日志表"""
    __tablename__ = "ai_detection_logs"
    
    log_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("call_records.call_id", ondelete="CASCADE"), nullable=False, index=True, comment="对应通话ID")
    
    # 三模态评分
    voice_confidence = Column(Float, default=0.0, comment="声音伪造置信度(0-1)")
    video_confidence = Column(Float, default=0.0, comment="图像伪造置信度(0-1)")
    text_confidence = Column(Float, default=0.0, comment="文本诈骗话术置信度(0-1)")
    
    overall_score = Column(Float, nullable=False, comment="综合风险评分(0-100)")
    detected_text = Column(Text, nullable=True, comment="检测到的完整文本内容")
    detected_keywords = Column(Text, nullable=True, comment="检测到的敏感关键词，如: 转账,验证码,屏幕共享")
    match_script = Column(String(100), nullable=True, comment="匹配的诈骗剧本")
    intent = Column(String(100), nullable=True, comment="识别的用户意图")
    evidence_snapshot = Column(String(500), nullable=True, comment="违规画面截图URL") 
    time_offset = Column(Integer, default=0, comment="异常发生的通话秒数")
    algorithm_details = Column(Text, nullable=True, comment="技术细节JSON，如: {face_swap: 0.9}") 
    model_version = Column(String(50), nullable=True, comment="使用的模型版本")
    
    # 图片OCR相关字段
    detection_type = Column(String(20), default="text", comment="检测类型: text/audio/video/image")
    image_ocr_text = Column(Text, nullable=True, comment="图片OCR提取的完整文字")
    ocr_dialogue_hash = Column(String(64), nullable=True, comment="对话内容哈希，用于去重")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="检测时间")
    
    def __repr__(self):
        return f"<AIDetectionLog(log_id={self.log_id}, call_id={self.call_id}, score={self.overall_score})>"