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
