"""
数据模型模块
"""
from .user import User
from .call_record import CallRecord, DetectionResult
from .ai_detection_log import AIDetectionLog
from .risk_rule import RiskRule
from .blacklist import NumberBlacklist

__all__ = [
    "User",
    "CallRecord",
    "DetectionResult",
    "AIDetectionLog",
    "RiskRule",
    "NumberBlacklist",
]
