"""
安全服务：负责风险规则匹配 (RiskRule)
"""
from sqlalchemy import select
# [修正] 这里原来写成了 async_session_maker，改为正确的 AsyncSessionLocal
from app.db.database import AsyncSessionLocal 
from app.models.risk_rule import RiskRule
from app.core.logger import get_logger

logger = get_logger(__name__)

class SecurityService:
    
    async def match_risk_rules(self, text: str) -> dict | None:
        """
        匹配文本风险规则 (数据库关键词)
        返回命中规则的详细信息，未命中返回 None
        """
        if not text:
            return None
            
        # [修正] 使用 AsyncSessionLocal() 创建会话
        async with AsyncSessionLocal() as db:
            try:
                # 查询所有启用的规则
                stmt = select(RiskRule).where(RiskRule.is_active == True)
                result = await db.execute(stmt)
                rules = result.scalars().all()
                
                for rule in rules:
                    if rule.keyword in text:
                        logger.warning(f"⚡ Risk Rule Hit: '{rule.keyword}' (ID: {rule.rule_id})")
                        return {
                            "rule_id": rule.rule_id,
                            "keyword": rule.keyword,
                            "risk_level": rule.risk_level,
                            "action": rule.action,
                            "description": rule.description
                        }
                return None
            except Exception as e:
                logger.error(f"Risk rule match failed: {e}")
                return None

# 全局实例
security_service = SecurityService()