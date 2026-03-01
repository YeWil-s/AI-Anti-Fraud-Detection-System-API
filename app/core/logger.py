"""
生产级日志配置模块 - Day 6 结构化升级版
支持 JSON 格式输出，方便 ELK/日志系统检索
支持上下文注入 (RequestID, UserID, CallID)
"""
import sys
import os
import logging
import json
import contextvars
from logging.handlers import TimedRotatingFileHandler
from typing import Union
from datetime import datetime

# 1. 定义上下文变量 (ContextVars)
request_id_ctx = contextvars.ContextVar("request_id", default="-")
user_id_ctx = contextvars.ContextVar("user_id", default="-")
call_id_ctx = contextvars.ContextVar("call_id", default="-")

# 2. [核心新增] 自定义 JSON 格式化器
class StructuredJsonFormatter(logging.Formatter):
    """
    将日志记录转化为结构化 JSON 字符串，方便机器解析
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
            # 注入上下文 ID
            "request_id": getattr(record, "request_id", "-"),
            "user_id": getattr(record, "user_id", "-"),
            "call_id": getattr(record, "call_id", "-")
        }
        # 如果有异常信息，也进行结构化捕获
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)


# 3. 上下文过滤器
class ContextFilter(logging.Filter):
    """
    将 ContextVars 中的变量注入到每一条日志记录中
    """
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        record.user_id = user_id_ctx.get()
        record.call_id = call_id_ctx.get()
        return True

# 4. 辅助函数：绑定上下文
def bind_context(
    user_id: Union[str, int, None] = None, 
    call_id: Union[str, int, None] = None
):
    if user_id is not None:
        user_id_ctx.set(str(user_id))
    if call_id is not None:
        call_id_ctx.set(str(call_id))

# 5. 初始化日志配置 
def setup_logging(level: str = "INFO"):
    """
    全局日志初始化配置
    """
    # 1. 确保日志目录存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. 初始化 JSON 格式化器和上下文过滤器
    formatter = StructuredJsonFormatter()
    ctx_filter = ContextFilter()

    # 3. 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 清除旧处理器
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # --- 处理器 1: 控制台输出 ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ctx_filter)
    root_logger.addHandler(console_handler)

    # --- 处理器 2: 文件输出 (按天切割) ---
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        when="midnight",  # 每天午夜切割
        interval=1,
        backupCount=30,   # 保留 30 天
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ctx_filter)
    root_logger.addHandler(file_handler)

    # 4. 屏蔽第三方库冗余日志
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)



# 6. 工厂函数与导出
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    return logger

# 预导出 logger
logger = get_logger("app")