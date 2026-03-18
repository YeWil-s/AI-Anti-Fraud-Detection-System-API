"""
数据模型模块
"""
from .family_group import FamilyGroup, FamilyApplication
from .user import User
from .call_record import CallRecord, DetectionResult
from .ai_detection_log import AIDetectionLog
from .risk_rule import RiskRule
from .blacklist import NumberBlacklist
from .message_log import MessageLog
from app.models.education import KnowledgeItem, UserLearningRecord
from .admin import Admin, AdminLog, SystemMonitor

__all__ = [
    "User",
    "CallRecord",
    "DetectionResult",
    "AIDetectionLog",
    "RiskRule",
    "NumberBlacklist",
    "MessageLog",
    "FamilyGroup",
    "FamilyApplication",
    "KnowledgeItem",
    "UserLearningRecord",
    "Admin",
    "AdminLog",
    "SystemMonitor",
]
