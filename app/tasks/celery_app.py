"""
Celery应用配置
"""
from celery import Celery
# [修正] 必须导入 crontab 才能使用定时任务调度
from celery.schedules import crontab  
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
    # 任务过期时间 (防止任务堆积)
    task_time_limit=1800,  # 30分钟
    worker_max_tasks_per_child=200,  # 防止内存泄漏
)

# 自动发现任务模块
celery_app.autodiscover_tasks([
    "app.tasks.detection_tasks",
    "app.tasks.maintenance_tasks"  # 确保这个文件存在
])

# [修正] 定时任务配置 (Celery Beat)
# 如果你还没有创建 maintenance_tasks.py，这段代码可能会导致 Celery 启动警告，
# 但不会导致主程序 main.py 崩溃（只要 crontab 导入了就行）。
celery_app.conf.beat_schedule = {
    # 每天凌晨3点清理30天前的旧日志
    'clean-old-logs-every-day': {
        'task': 'app.tasks.maintenance_tasks.clean_old_logs',
        'schedule': crontab(hour=3, minute=0),  # <--- 这里就是报错的地方
        'args': (30,)  # 保留30天数据
    },
}