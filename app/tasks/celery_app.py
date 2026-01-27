"""
Celery应用配置
"""
from celery import Celery
from app.core.config import settings

# 创建Celery应用实例
celery_app = Celery(
    "ai_fraud_detection",
    broker=settings.CELERY_BROKER_URL,  # Redis作为消息代理
    backend=settings.CELERY_RESULT_BACKEND  # Redis作为结果后端
)

# Celery配置
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,  # 追踪任务状态
    task_time_limit=30 * 60,  # 任务超时时间30分钟
    worker_prefetch_multiplier=4,  # Worker预取任务数
    worker_max_tasks_per_child=200,  # Worker最大任务数后重启
)

# 自动发现任务
celery_app.autodiscover_tasks(['app.tasks'])

if __name__ == '__main__':
    celery_app.start()
