"""
app/api/admin.py
管理员专用接口：异步版本
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from typing import List

from app.db.database import get_db
from app.models.risk_rule import RiskRule
from app.models.blacklist import NumberBlacklist
from app.models.user import User
from app.models.call_record import CallRecord, DetectionResult
from app.schemas.admin import (
    RiskRuleCreate, RiskRuleUpdate, RiskRuleResponse,
    BlacklistCreate, BlacklistUpdate, BlacklistResponse
)

router = APIRouter()

# =======================
# 1. 仪表盘数据统计 (异步重写)
# =======================
@router.get("/stats", summary="获取仪表盘统计数据")
async def get_admin_stats(db: AsyncSession = Depends(get_db)):
    # 1. 总用户数
    result = await db.execute(select(func.count(User.user_id)))
    total_users = result.scalar() or 0
    
    # 2. 总通话记录数
    result = await db.execute(select(func.count(CallRecord.call_id)))
    total_calls = result.scalar() or 0
    
    # 3. 累计拦截诈骗
    result = await db.execute(
        select(func.count(CallRecord.call_id)).where(CallRecord.detected_result == DetectionResult.FAKE)
    )
    fraud_calls = result.scalar() or 0
    
    # 4. 黑名单号码数
    result = await db.execute(select(func.count(NumberBlacklist.id)))
    blacklist_count = result.scalar() or 0

    # 5. 风险规则数
    result = await db.execute(select(func.count(RiskRule.rule_id)))
    rule_count = result.scalar() or 0

    return {
        "total_users": total_users,
        "total_calls": total_calls,
        "fraud_blocked": fraud_calls,
        "blacklist_count": blacklist_count,
        "active_rules": rule_count,
        "system_health": "100%"
    }

# =======================
# 2. 功能测试台 (异步重写)
# =======================
@router.post("/test/text_match", summary="测试文本规则匹配")
async def test_text_rule_match(text: str, db: AsyncSession = Depends(get_db)):
    # 获取所有启用规则
    result = await db.execute(select(RiskRule).where(RiskRule.is_active == True))
    rules = result.scalars().all()
    
    hits = []
    max_risk = 0
    action = "pass"
    
    for rule in rules:
        if rule.keyword in text:
            hits.append(rule.keyword)
            if rule.risk_level > max_risk:
                max_risk = rule.risk_level
                # 如果有 block 规则命中，直接升级为 block
                if rule.action == "block":
                    action = "block"
    
    # 如果没被 block 但有风险，设为 alert
    if action != "block" and max_risk > 0:
        action = "alert"
    
    return {
        "text_length": len(text),
        "hit_keywords": hits,
        "risk_level": max_risk,
        "action": action
    }

# =======================
# 3. 风险规则管理 (异步 CRUD)
# =======================
@router.get("/rules", response_model=List[RiskRuleResponse])
async def get_risk_rules(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RiskRule).offset(skip).limit(limit))
    return result.scalars().all()

@router.post("/rules", response_model=RiskRuleResponse)
async def create_risk_rule(rule: RiskRuleCreate, db: AsyncSession = Depends(get_db)):
    # 查重
    result = await db.execute(select(RiskRule).where(RiskRule.keyword == rule.keyword))
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail="关键词已存在")
    
    db_rule = RiskRule(**rule.dict())
    db.add(db_rule)
    await db.commit()
    await db.refresh(db_rule)
    return db_rule

@router.delete("/rules/{rule_id}")
async def delete_risk_rule(rule_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RiskRule).where(RiskRule.rule_id == rule_id))
    db_rule = result.scalar_one_or_none()
    
    if not db_rule:
        raise HTTPException(404, "规则不存在")
    
    await db.delete(db_rule)
    await db.commit()
    return {"msg": "Deleted"}

# =======================
# 4. 黑名单管理 (异步 CRUD)
# =======================
@router.get("/blacklist", response_model=List[BlacklistResponse])
async def get_blacklist(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NumberBlacklist).offset(skip).limit(limit))
    return result.scalars().all()

@router.post("/blacklist", response_model=BlacklistResponse)
async def add_blacklist(item: BlacklistCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NumberBlacklist).where(NumberBlacklist.number == item.number))
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(400, "号码已在黑名单")
    
    db_item = NumberBlacklist(**item.dict())
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@router.delete("/blacklist/{id}")
async def remove_blacklist(id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(NumberBlacklist).where(NumberBlacklist.id == id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(404, "记录不存在")
    
    await db.delete(item)
    await db.commit()
    return {"msg": "Deleted"}