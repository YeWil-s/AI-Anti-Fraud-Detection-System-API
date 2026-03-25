"""
长期记忆服务 (Long-term Memory)
管理用户的跨通话历史记忆，用于个性化风控决策
"""
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime

from app.models.user_memory import UserMemory, MemoryType
from app.models.user import User
from app.core.logger import get_logger

logger = get_logger(__name__)


class LongTermMemoryService:
    """长期记忆服务"""
    
    async def add_memory(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: str,
        content: str,
        importance: int = 3,
        source_call_id: Optional[int] = None
    ) -> Optional[UserMemory]:
        """
        添加新的长期记忆
        
        Args:
            user_id: 用户ID
            memory_type: 记忆类型 (fraud_experience/alert_response/preference/risk_pattern)
            content: 记忆内容
            importance: 重要性 1-5
            source_call_id: 来源通话ID（可选）
        """
        # 检查是否已存在相似记忆
        existing = await db.execute(
            select(UserMemory).where(
                and_(
                    UserMemory.user_id == user_id,
                    UserMemory.memory_type == memory_type,
                    UserMemory.content == content
                )
            )
        )
        if existing.scalar_one_or_none():
            logger.info(f"相似记忆已存在，跳过添加: user={user_id}, type={memory_type}")
            return None
        
        memory = UserMemory(
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            importance=min(max(importance, 1), 5),  # 限制在1-5范围
            source_call_id=source_call_id
        )
        db.add(memory)
        await db.commit()
        await db.refresh(memory)
        
        logger.info(f"添加长期记忆: user={user_id}, type={memory_type}, importance={importance}")
        return memory
    
    async def get_memories(
        self,
        db: AsyncSession,
        user_id: int,
        memory_type: Optional[str] = None,
        min_importance: int = 1,
        limit: int = 10
    ) -> List[UserMemory]:
        """
        获取用户的长期记忆
        
        Args:
            user_id: 用户ID
            memory_type: 过滤特定类型（可选）
            min_importance: 最小重要性
            limit: 返回数量限制
        """
        query = select(UserMemory).where(
            and_(
                UserMemory.user_id == user_id,
                UserMemory.importance >= min_importance
            )
        )
        
        if memory_type:
            query = query.where(UserMemory.memory_type == memory_type)
        
        query = query.order_by(UserMemory.importance.desc(), UserMemory.created_at.desc()).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_memory_summary(self, db: AsyncSession, user_id: int) -> str:
        """
        获取用户的记忆摘要（用于Prompt）
        
        Returns:
            格式化的记忆摘要文本
        """
        # 优先使用预生成的摘要
        user_result = await db.execute(select(User).where(User.user_id == user_id))
        user = user_result.scalar_one_or_none()
        
        if user and user.memory_summary:
            return user.memory_summary
        
        # 动态生成摘要
        memories = await self.get_memories(db, user_id, limit=5)
        if not memories:
            return "该用户暂无历史记忆。"
        
        summary_parts = []
        for m in memories:
            type_name = {
                MemoryType.FRAUD_EXPERIENCE.value: "诈骗经历",
                MemoryType.ALERT_RESPONSE.value: "告警反应",
                MemoryType.PREFERENCE.value: "个人偏好",
                MemoryType.RISK_PATTERN.value: "风险模式",
                MemoryType.LEARNING_PROGRESS.value: "学习进度"
            }.get(m.memory_type, m.memory_type)
            
            summary_parts.append(f"- [{type_name}] {m.content}")
        
        return "\n".join(summary_parts)
    
    async def update_memory_summary(self, db: AsyncSession, user_id: int):
        """
        更新用户的记忆摘要（定期执行）
        """
        memories = await self.get_memories(db, user_id, limit=20)
        if not memories:
            return
        
        # 按类型分组统计
        fraud_experiences = [m for m in memories if m.memory_type == MemoryType.FRAUD_EXPERIENCE.value]
        alert_responses = [m for m in memories if m.memory_type == MemoryType.ALERT_RESPONSE.value]
        risk_patterns = [m for m in memories if m.memory_type == MemoryType.RISK_PATTERN.value]
        
        summary_parts = ["【用户历史行为摘要】"]
        
        # 诈骗经历总结
        if fraud_experiences:
            fraud_types = {}
            for m in fraud_experiences:
                # 从内容中提取诈骗类型
                if "刷单" in m.content:
                    fraud_types["刷单返利"] = fraud_types.get("刷单返利", 0) + 1
                elif "投资" in m.content or "理财" in m.content:
                    fraud_types["虚假投资"] = fraud_types.get("虚假投资", 0) + 1
                elif "公检法" in m.content:
                    fraud_types["冒充公检法"] = fraud_types.get("冒充公检法", 0) + 1
                elif "杀猪盘" in m.content or "网恋" in m.content:
                    fraud_types["杀猪盘"] = fraud_types.get("杀猪盘", 0) + 1
                else:
                    fraud_types["其他"] = fraud_types.get("其他", 0) + 1
            
            summary_parts.append(f"遭遇诈骗: {len(fraud_experiences)}次")
            for ft, count in sorted(fraud_types.items(), key=lambda x: -x[1]):
                summary_parts.append(f"  - {ft}: {count}次")
        
        # 告警反应总结
        if alert_responses:
            obey_count = sum(1 for m in alert_responses if "听从" in m.content or "接受" in m.content)
            ignore_count = sum(1 for m in alert_responses if "忽略" in m.content or "拒绝" in m.content)
            summary_parts.append(f"告警反应: 听从{obey_count}次, 忽略{ignore_count}次")
        
        # 风险模式
        if risk_patterns:
            summary_parts.append(f"风险特征: {len(risk_patterns)}项")
            for m in risk_patterns[:3]:
                summary_parts.append(f"  - {m.content}")
        
        summary = "\n".join(summary_parts)
        
        # 更新到用户表
        user_result = await db.execute(select(User).where(User.user_id == user_id))
        user = user_result.scalar_one_or_none()
        if user:
            user.memory_summary = summary
            await db.commit()
            logger.info(f"更新用户记忆摘要: user={user_id}")
        
        return summary
    
    async def extract_and_save_memories(
        self,
        db: AsyncSession,
        user_id: int,
        call_id: int,
        detection_result: dict,
        user_response: Optional[str] = None
    ):
        """
        从检测结果中提取并保存记忆
        
        Args:
            detection_result: LLM的检测结果
            user_response: 用户对告警的反应（可选）
        """
        memories_added = []
        
        # 1. 保存诈骗经历
        if detection_result.get("is_fraud"):
            fraud_type = detection_result.get("fraud_type", "未知类型")
            match_script = detection_result.get("match_script", "")
            
            content = f"遭遇{fraud_type}"
            if match_script and match_script != "无":
                content += f"，匹配剧本：{match_script}"
            
            memory = await self.add_memory(
                db=db,
                user_id=user_id,
                memory_type=MemoryType.FRAUD_EXPERIENCE.value,
                content=content,
                importance=4,  # 诈骗经历重要性较高
                source_call_id=call_id
            )
            if memory:
                memories_added.append(memory)
        
        # 2. 保存用户对告警的反应
        if user_response:
            memory = await self.add_memory(
                db=db,
                user_id=user_id,
                memory_type=MemoryType.ALERT_RESPONSE.value,
                content=f"系统告警后用户反应：{user_response}",
                importance=3,
                source_call_id=call_id
            )
            if memory:
                memories_added.append(memory)
        
        # 3. 提取风险行为模式
        risk_level = detection_result.get("risk_level", "safe")
        if risk_level in ["suspicious", "fake", "high", "critical"]:
            intent = detection_result.get("intent", "")
            if intent and "转账" in intent:
                memory = await self.add_memory(
                    db=db,
                    user_id=user_id,
                    memory_type=MemoryType.RISK_PATTERN.value,
                    content="易被诱导转账",
                    importance=4,
                    source_call_id=call_id
                )
                if memory:
                    memories_added.append(memory)
        
        # 4. 更新摘要
        if memories_added:
            await self.update_memory_summary(db, user_id)
        
        return memories_added
    
    async def get_context_for_prompt(self, db: AsyncSession, user_id: int) -> str:
        """
        获取用于Prompt的记忆上下文
        
        Returns:
            格式化的记忆文本
        """
        summary = await self.get_memory_summary(db, user_id)
        
        if summary == "该用户暂无历史记忆。":
            return ""
        
        return f"""
【用户历史记忆（跨通话长期记忆）】
{summary}

【基于历史记忆的审查建议】
- 如果用户有多次忽略告警的记录，本次应加强告警强度
- 如果用户曾遭遇某类诈骗，对相似话术应提高敏感度
- 结合用户画像和历史行为，动态调整风险阈值
"""


# 全局单例
long_term_memory_service = LongTermMemoryService()
