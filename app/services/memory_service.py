"""
对话记忆服务 (Short-term Memory Pool)
利用 Redis 为每通电话维护一个滑动窗口的上下文历史
"""
import redis
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class MemoryService:
    def __init__(self):
        # 使用普通的同步 Redis 客户端即可，速度极快
        self.redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.max_history = 8  # 默认记住最近的 8 句话 (足够覆盖一个完整的诈骗话术连招)

    def add_message(self, call_id: int, text: str):
        """将新识别的文本加入该通话的记忆池"""
        if not text.strip():
            return
            
        key = f"chat_history:{call_id}"
        
        try:
            # 1. 存入 Redis 列表的最右侧
            self.redis_client.rpush(key, text)
            
            # 2. 截断列表，只保留最新的 max_history 条
            self.redis_client.ltrim(key, -self.max_history, -1)
            
            # 3. 刷新过期时间（通话结束后1小时自动销毁，保护隐私）
            self.redis_client.expire(key, 3600)
            
        except Exception as e:
            logger.error(f"Failed to add message to memory pool: {e}")

    def get_context(self, call_id: int) -> str:
        """获取并拼接近期的上下文内容"""
        key = f"chat_history:{call_id}"
        try:
            history = self.redis_client.lrange(key, 0, -1)
            if not history:
                return "暂无历史上下文。"
            
            # 将历史记录用换行符拼接起来，标上序号
            formatted_history = "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(history)])
            return formatted_history
        except Exception as e:
            logger.error(f"Failed to fetch memory context: {e}")
            return "暂无历史上下文。"

# 全局单例
memory_service = MemoryService()