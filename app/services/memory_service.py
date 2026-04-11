"""
对话记忆服务 (Short-term Memory Pool)
利用 Redis 为每通电话维护一个滑动窗口的上下文历史
"""
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class MemoryService:
    def __init__(self):
        self.max_history = 8  # 默认记住最近的 8 句话

    def _create_client(self):
        """
        为每次调用创建独立 Redis 客户端，避免 Celery 线程池下跨事件循环复用导致
        `Event loop is closed` 或 `attached to a different loop`。
        """
        return aioredis.from_url(settings.REDIS_URL, decode_responses=True)

    async def add_message(self, call_id: int, text: str):
        """将新识别的文本加入该通话的记忆池"""
        if not text.strip():
            return

        key = f"chat_history:{call_id}"
        client = self._create_client()
        try:
            await client.rpush(key, text)
            await client.ltrim(key, -self.max_history, -1)
            await client.expire(key, 3600)
        except Exception as e:
            logger.error(f"Failed to add message to memory pool: {e}")
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    async def get_context(self, call_id: int) -> str:
        """获取并拼接近期的上下文内容"""
        key = f"chat_history:{call_id}"
        client = self._create_client()
        try:
            history = await client.lrange(key, 0, -1)
            if not history:
                return "暂无历史上下文。"

            return "\n".join([f"{i + 1}. {msg}" for i, msg in enumerate(history)])
        except Exception as e:
            logger.error(f"Failed to get context from memory pool: {e}")
            return "获取上下文失败。"
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    async def clear_context(self, call_id: int):
        """通话结束后清空该通话的上下文"""
        key = f"chat_history:{call_id}"
        client = self._create_client()
        try:
            await client.delete(key)
        except Exception as e:
            logger.error(f"Failed to clear memory pool: {e}")
        finally:
            try:
                await client.aclose()
            except Exception:
                pass


# 实例化单例
memory_service = MemoryService()