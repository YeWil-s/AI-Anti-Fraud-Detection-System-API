"""
Celery应用配置 
"""
import os
import sys
from unittest.mock import MagicMock
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.modules['chromadb.telemetry.posthog'] = MagicMock()
sys.modules['posthog'] = MagicMock()

from celery import Celery
from celery.schedules import crontab  #  导入定时调度工具
from app.core.config import settings

# 初始化Celery应用
celery_app = Celery(
    "fraud_detection",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# 配置Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=False,
    # 任务过期时间
    task_time_limit=1800,  # 30分钟
    worker_max_tasks_per_child=200,  # 防止内存泄漏
)

# 自动发现任务模块
# 建议直接指向任务所在的文件夹包名
celery_app.autodiscover_tasks(["app.tasks"])

# 定时任务配置 (Celery Beat)
celery_app.conf.beat_schedule = {
    # 每天凌晨3点清理30天前的旧日志
    'clean-old-logs-every-day': {
        'task': 'clean_old_logs', 
        'schedule': crontab(hour=3, minute=0),
        'args': (30,) 
    },
    # 每天凌晨4点自动扫描并学习新型诈骗案例
    'auto-learn-new-cases-every-day': {
        'task': 'auto_learn_new_cases', 
        'schedule': crontab(hour=4, minute=0), # 安排在日志清理之后执行
        'args': ()
    },
}