"""
Redis连接工具
"""
import redis.asyncio as aioredis
from typing import Optional
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# 全局Redis连接池
redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """获取Redis连接 (单例模式)"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                # 添加连接池自动回收和心跳检测，增加稳定性
                socket_keepalive=True,
                health_check_interval=30
            )
            # 尝试 ping 一下确保连接通畅
            await redis_client.ping()
            logger.info("📡 Redis connection established.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            redis_client = None
            raise
    return redis_client


async def close_redis():
    """关闭Redis连接"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        logger.info("📡 Redis connection closed.")

async def set_user_preference(user_id: int, key: str, value: str, expire: int = 86400):
    """设置用户偏好"""
    # 始终通过 get_redis 获取，确保连接已初始化
    redis = await get_redis()
    await redis.hset(f"users:{user_id}:prefs", key, value)
    await redis.expire(f"users:{user_id}:prefs", expire)

async def get_user_preference(user_id: int, key: str, default: str = None) -> str:
    """获取用户偏好"""
    redis = await get_redis()
    value = await redis.hget(f"users:{user_id}:prefs", key)
    return value if value else default

async def get_all_user_preferences(user_id: int) -> dict:
    """获取用户所有配置"""
    redis = await get_redis()
    return await redis.hgetall(f"users:{user_id}:prefs")