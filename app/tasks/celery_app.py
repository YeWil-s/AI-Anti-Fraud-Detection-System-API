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
    enable_utc=False,
    task_track_started=True,  # 追踪任务状态
    task_time_limit=30 * 60,  # 任务超时时间30分钟
    worker_prefetch_multiplier=4,  # Worker预取任务数
    worker_max_tasks_per_child=200,  # Worker最大任务数后重启
)

# 定时任务调度配置 (Celery Beat)
celery_app.conf.beat_schedule = {
    # 每天凌晨 3:00 执行一次清理任务
    'clean-db-logs-every-night': {
        'task': 'clean_old_logs',          # 任务名称 (对应 maintenance_tasks.py 中的 name)
        'schedule': crontab(hour=3, minute=0), 
        'args': (30,)                      # 参数: 保留 30 天数据
    },
}

# 自动发现任务
celery_app.autodiscover_tasks(['app.tasks.detection_tasks', 'app.tasks.maintenance_tasks'])

if __name__ == '__main__':
    celery_app.start()
