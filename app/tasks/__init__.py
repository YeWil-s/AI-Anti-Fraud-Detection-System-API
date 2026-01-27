"""
Celery异步任务模块
"""
from app.tasks.celery_app import celery_app
from app.tasks.detection_tasks import detect_audio_task, detect_video_task, detect_text_task

__all__ = [
    "celery_app",
    "detect_audio_task",
    "detect_video_task",
    "detect_text_task"
]
