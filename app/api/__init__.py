"""
API路由模块
"""
from .users import router as users_router
from .detection import router as detection_router
from .tasks import router as tasks_router
from .call_records import router as call_records_router

__all__ = ["users_router", "detection_router", "tasks_router", "call_records_router"]
