"""
FastAPI主应用入口
"""
import os
import sys
from unittest.mock import MagicMock

# 禁用所有遥测/统计功能（必须在导入ChromaDB之前设置）
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "true"

# 创建一个模拟的telemetry模块，让它返回可调用对象
class MockTelemetry:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def capture(self, *args, **kwargs):
        # ChromaDB 可能以多种方式调用 capture，需要兼容所有情况
        return self
    def identify(self, *args, **kwargs):
        return self
    def alias(self, *args, **kwargs):
        return self
    def group(self, *args, **kwargs):
        return self

# 创建完整的 Mock posthog 模块
mock_posthog = MagicMock()
mock_posthog.capture = MockTelemetry().capture
mock_posthog.identify = MockTelemetry().identify
mock_posthog.alias = MockTelemetry().alias
mock_posthog.group = MockTelemetry().group

# Mock posthog 相关模块
sys.modules['posthog'] = mock_posthog
sys.modules['chromadb.telemetry.posthog'] = MagicMock()

# 同时 Mock chromadb 的 telemetry 相关模块
try:
    import chromadb
    if hasattr(chromadb, 'telemetry'):
        sys.modules['chromadb.telemetry'] = MagicMock()
except ImportError:
    pass
import asyncio
import json
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db.database import engine
from app.api import users_router, detection_router, tasks_router, call_records_router, family
from app.api.education import router as education_router
from app.api.admin import router as admin_router
from app.core.config import settings
from app.core.logger import setup_logging, logger, request_id_ctx
from app.core.redis import get_redis, close_redis 
from app.db.database import init_db
from app.services.websocket_manager import connection_manager

# Redis 监听服务 (桥梁)
async def redis_listener():
    """
    后台任务：监听 Redis 消息并转发给 WebSocket
    """
    pubsub = None
    try:
        redis = await get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe("fraud_alerts")
        
        logger.info("Redis 消息监听器已启动: 监听频道 [fraud_alerts]")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    user_id = data.get('user_id')
                    payload = data.get('payload')
                    
                    if user_id and connection_manager.is_user_online(user_id):
                        # control+upgrade_level → 仅下发 level_sync（静默同步防御等级与采集策略，不弹诈骗窗）
                        # detection_result 等原样转发；融合判定高危由任务发 type=alert（与前端弹窗一致）
                        if payload.get("type") == "control" and payload.get("action") == "upgrade_level":
                            target_level = payload.get("target_level")
                            config = payload.get("config")
                            await connection_manager.set_defense_level(user_id, target_level, config)
                        else:
                            await connection_manager.send_personal_message(payload, user_id)
                        
                        logger.info(f" [转发成功] Celery -> User {user_id} | Type: {payload.get('type')}")
                    else:
                        logger.debug(f"用户 {user_id} 不在线，消息丢弃")
                        
                except Exception as e:
                    logger.error(f"消息转发解析异常: {e}")
                    
    except asyncio.CancelledError:
        logger.info("Redis 监听任务正在取消...")
    except Exception as e:
        logger.error(f"Redis 监听器致命错误: {e}")
    finally:
        if pubsub:
            await pubsub.close()

# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动与关闭的勾子函数"""
    # --- 启动阶段 ---
    setup_logging(level="INFO" if not settings.DEBUG else "DEBUG")
    logger.info("应用正在启动...")
    
    # 1. 初始化数据库
    try:
        await init_db()
        logger.info("数据库初始化完成")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")

    # 2. 启动 WebSocket 心跳检查 
    heartbeat_task = asyncio.create_task(connection_manager.heartbeat_check(interval=30))
    
    # 3. 启动 Redis 警报监听器
    listener_task = asyncio.create_task(redis_listener())
    
    yield
    
    # --- 关闭阶段 ---
    logger.info("应用正在关闭...")
    
    # 取消所有后台任务
    heartbeat_task.cancel()
    listener_task.cancel()
    
    # 等待任务清理完毕
    await asyncio.gather(heartbeat_task, listener_task, return_exceptions=True)
    
    # 关闭 Redis 连接池
    await close_redis()
    logger.info("应用已完全停止")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,  
    version=settings.APP_VERSION,
    description="AI伪造检测与诈骗预警系统后端API",
    lifespan=lifespan
)

# 中间件与路由配置
@app.middleware("http")
async def logger_middleware(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    token = request_id_ctx.set(req_id)
    logger.info(f"请求开始: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"请求结束: status={response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"请求处理异常: {e}")
        raise
    finally:
        request_id_ctx.reset(token)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(detection_router)
app.include_router(tasks_router)
app.include_router(call_records_router)
app.include_router(family.router)
app.include_router(admin_router, prefix="/api/admin", tags=["Admin Management"])
app.include_router(education_router)

@app.get("/")
async def root():
    return {
        "message": "AI Anti-Fraud Detection System API",
        "version": settings.APP_VERSION,
        "status": "running"
    }

@app.get("/health", tags=["System"])
async def health_check():
    """全链路健康检查：验证数据库和 Redis 状态"""
    db_status = "down"
    redis_status = "down"
    
    # 1. 检查数据库
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "up"
    except Exception as e:
        logger.error(f"Health check: DB connection failed: {e}")

    # 2. 检查 Redis
    try:
        redis = await get_redis()
        await redis.ping()
        redis_status = "up"
    except Exception as e:
        logger.error(f"Health check: Redis connection failed: {e}")

    overall_status = "healthy" if db_status == "up" and redis_status == "up" else "degraded"
    
    return {
        "status": overall_status,
        "details": {
            "database": db_status,
            "redis": redis_status,
            "version": settings.APP_VERSION
        }
    }

# 导出 app 供 ASGI 服务器使用
__all__ = ["app"]

if __name__ == "__main__":
    import platform
    
    if platform.system() == "Windows":
        # Windows 下直接传入 app 对象，避免模块重新导入问题
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # 其他系统可以使用 reload 模式
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)