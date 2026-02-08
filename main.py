"""
FastAPI主应用入口
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uuid
import uvicorn
import asyncio
import json
from redis import asyncio as aioredis  # [新增] 异步 Redis 客户端

from app.core.config import settings
from app.db.database import init_db
from app.api import users_router, detection_router, tasks_router, call_records_router
from app.services.websocket_manager import connection_manager  # [新增] 导入连接管理器

from app.core.logger import setup_logging, logger, request_id_ctx

# =========================================================
# [新增] Redis 监听服务 (核心桥梁)
# =========================================================
async def redis_listener():
    """
    后台任务：监听 Redis 消息并转发给 WebSocket
    这是 Celery (独立进程) 和 FastAPI (主进程) 之间的传声筒。
    """
    redis = None
    pubsub = None
    try:
        # 创建异步 Redis 连接
        redis = aioredis.from_url(settings.REDIS_URL)
        pubsub = redis.pubsub()
        await pubsub.subscribe("fraud_alerts")
        
        logger.info("🎧 Redis 消息监听器已启动: 监听频道 [fraud_alerts]")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    # 1. 解析 Celery 发过来的数据
                    # 数据格式: {"user_id": 123, "payload": {...}}
                    data = json.loads(message['data'])
                    user_id = data.get('user_id')
                    payload = data.get('payload')
                    
                    # 2. 转发给 WebSocket
                    # 因为这个函数运行在 Main 进程，它能访问到真正的 active_connections
                    if user_id and connection_manager.is_user_online(user_id):
                        if payload.get("type") == "control" and payload.get("action") == "upgrade_level":
                            target_level = payload.get("target_level")
                            config = payload.get("config")
                            await connection_manager.set_defense_level(user_id, target_level, config)
                        else:
                            await connection_manager.send_personal_message(payload, user_id)
                        logger.info(f"📡 [转发成功] Celery -> User {user_id} | Type: {payload.get('type')}")
                    else:
                        # 用户可能已经断开了，这是正常现象
                        logger.debug(f"用户 {user_id} 不在线，消息丢弃")
                        
                except Exception as e:
                    logger.error(f"消息转发异常: {e}")
                    
    except asyncio.CancelledError:
        logger.info("Redis 监听任务被取消")
    except Exception as e:
        logger.error(f"Redis 监听器致命错误: {e}")
    finally:
        if pubsub: await pubsub.close()
        if redis: await redis.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 1. 初始化日志系统
    setup_logging(level="INFO" if not settings.DEBUG else "DEBUG")
    
    # 2. [新增] 启动 Redis 监听器 (后台运行)
    # create_task 会让它在后台跑，不会阻塞主线程启动
    listener_task = asyncio.create_task(redis_listener())
    
    logger.info("🚀 应用正在启动...")
    try:
        await init_db()
        logger.info("✅ 数据库初始化完成")
    except Exception as e:
        logger.error(f"❌ 数据库连接失败: {e}")
    
    yield
    
    # 3. [新增] 关闭时清理后台任务
    logger.info("🛑 应用正在关闭...")
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,  
    version=settings.APP_VERSION,
    description="AI伪造检测与诈骗预警系统后端API",
    lifespan=lifespan
)


# =========================================================
# 中间件：注入 Request ID
# =========================================================
@app.middleware("http")
async def logger_middleware(request: Request, call_next):
    # 1. 生成 8位 唯一请求ID
    req_id = str(uuid.uuid4())[:8]
    
    # 2. 设置到 ContextVar
    token = request_id_ctx.set(req_id)
    
    logger.info(f"➡️ 请求开始: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        logger.info(f"⬅️ 请求结束: status={response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"❌ 请求处理异常: {e}")
        raise
    finally:
        # 3. 重置上下文
        request_id_ctx.reset(token)


# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(users_router)
app.include_router(detection_router)
app.include_router(tasks_router)
app.include_router(call_records_router)


@app.get("/")
async def root():
    """根路径"""
    logger.info("访问了根路径") 
    return {
        "message": "AI Anti-Fraud Detection System API",
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)