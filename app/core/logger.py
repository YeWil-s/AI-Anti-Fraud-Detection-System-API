"""
生产级日志配置模块
支持上下文注入 (RequestID, UserID, CallID)
支持自动生成 logs 文件夹并按天切割日志
"""
import sys
import os  # [新增] 需要用到 os 模块
import logging
import contextvars
from logging.handlers import TimedRotatingFileHandler # [新增] 用于按天切割日志
from typing import Union

# ==========================================
# 1. 定义上下文变量 (ContextVars)
# ==========================================
request_id_ctx = contextvars.ContextVar("request_id", default="-")
user_id_ctx = contextvars.ContextVar("user_id", default="-")
call_id_ctx = contextvars.ContextVar("call_id", default="-")


# ==========================================
# 2. 上下文过滤器
# ==========================================
class ContextFilter(logging.Filter):
    """
    将 ContextVars 中的变量注入到每一条日志记录中
    """
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        record.user_id = user_id_ctx.get()
        record.call_id = call_id_ctx.get()
        return True


# ==========================================
# 3. 辅助函数：绑定上下文
# ==========================================
def bind_context(
    user_id: Union[str, int, None] = None, 
    call_id: Union[str, int, None] = None
):
    if user_id is not None:
        user_id_ctx.set(str(user_id))
    if call_id is not None:
        call_id_ctx.set(str(call_id))


# ==========================================
# 4. 初始化日志配置 (核心修改部分)
# ==========================================
def setup_logging(level: str = "INFO"):
    """
    全局日志初始化配置
    """
    # [新增] 1. 确保日志目录存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"✅ 已自动创建日志文件夹: {os.path.abspath(log_dir)}")

    # 定义日志格式
    log_format = (
        "%(asctime)s | %(levelname)-7s | "
        "req:%(request_id)s | uid:%(user_id)s | cid:%(call_id)s | "
        "%(name)s:%(lineno)d - %(message)s"
    )
    formatter = logging.Formatter(log_format)
    ctx_filter = ContextFilter()

    # 2. 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 3. 清除已有的处理器
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # --- 处理器 1: 控制台输出 ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ctx_filter)
    root_logger.addHandler(console_handler)

    # --- [新增] 处理器 2: 文件输出 (按天切割) ---
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        when="midnight",  # 每天午夜切割
        interval=1,       # 间隔 1 天
        backupCount=30,   # 保留 30 天
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ctx_filter) # 记得给文件也加上过滤器
    root_logger.addHandler(file_handler)

    # 5. 调整第三方库的日志级别
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# ==========================================
# 5. 获取 Logger 实例的工厂函数
# ==========================================
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    return logger

# 预导出一个默认 logger
logger = get_logger("app")