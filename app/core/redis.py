"""
Redis连接工具
"""
import redis.asyncio as aioredis
from typing import Optional
from app.core.config import settings

# 全局Redis连接池
redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """获取Redis连接"""
    global redis_client
    if redis_client is None:
        redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    return redis_client


async def close_redis():
    """关闭Redis连接"""
    global redis_client
    if redis_client:
        await redis_client.close()

async def set_user_preference(user_id: int, key: str, value: str, expire: int = 86400):
    """设置用户偏好 (如 fps, sensitivity)"""
    redis = await get_redis()
    # 使用 Hash 结构: users:123:prefs -> { "fps": "5", "risk_level": "high" }
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