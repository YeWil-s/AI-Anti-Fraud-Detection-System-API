"""
生产级日志配置模块
支持上下文注入 (RequestID, UserID, CallID)
"""
import sys
import logging
import contextvars
import uuid
from typing import Optional, Union

# ==========================================
# 1. 定义上下文变量 (ContextVars)
# ==========================================
# 这些变量是协程安全的，用于在异步任务链中传递上下文信息
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
    """
    手动绑定用户ID或通话ID到当前上下文
    通常在路由处理函数或 Service 层中使用
    """
    if user_id is not None:
        user_id_ctx.set(str(user_id))
    if call_id is not None:
        call_id_ctx.set(str(call_id))


# ==========================================
# 4. 初始化日志配置
# ==========================================
def setup_logging(level: str = "INFO"):
    """
    全局日志初始化配置
    """
    # 定义日志格式
    # 格式示例: 2026-01-15 10:30:01 | INFO | req:a1b2 | uid:1001 | cid:55 | model_service:45 - 模型加载成功
    log_format = (
        "%(asctime)s | %(levelname)-7s | "
        "req:%(request_id)s | uid:%(user_id)s | cid:%(call_id)s | "
        "%(name)s:%(lineno)d - %(message)s"
    )

    # 1. 获取根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 2. 清除已有的处理器 (防止重复日志，例如 uvicorn 默认的)
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # 3. 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 4. 添加上下文过滤器
    console_filter = ContextFilter()
    console_handler.addFilter(console_filter)
    
    root_logger.addHandler(console_handler)

    # 5. 调整第三方库的日志级别 (减少噪音)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# ==========================================
# 5. 获取 Logger 实例的工厂函数
# ==========================================
def get_logger(name: str) -> logging.Logger:
    """
    获取带上下文过滤器的 logger
    """
    logger = logging.getLogger(name)
    # 确保 logger 也会使用 filter (虽然 root logger 已经配置，但为了保险)
    if not any(isinstance(f, ContextFilter) for f in logger.filters):
        logger.addFilter(ContextFilter())
    return logger

# 预导出一个默认 logger
logger = get_logger("app")