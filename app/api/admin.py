from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.models.risk_rule import RiskRule
from app.models.blacklist import NumberBlacklist
# 引入刚才创建的 schemas
from app.schemas.admin import (
    RiskRuleCreate, RiskRuleUpdate, RiskRuleResponse,
    BlacklistCreate, BlacklistUpdate, BlacklistResponse
)

# 定义路由，统一前缀在 main.py 中配置，这里留空或设为 ""
router = APIRouter()

# =======================
# 1. 风险规则管理 (Risk Rules)
# =======================

@router.get("/rules", response_model=List[RiskRuleResponse], summary="获取所有风险规则")
def get_risk_rules(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    rules = db.query(RiskRule).offset(skip).limit(limit).all()
    return rules

@router.post("/rules", response_model=RiskRuleResponse, summary="添加新的风险关键词")
def create_risk_rule(rule: RiskRuleCreate, db: Session = Depends(get_db)):
    # 检查关键词是否已存在
    existing = db.query(RiskRule).filter(RiskRule.keyword == rule.keyword).first()
    if existing:
        raise HTTPException(status_code=400, detail="该关键词已存在")
    
    db_rule = RiskRule(**rule.dict())
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    return db_rule

@router.put("/rules/{rule_id}", response_model=RiskRuleResponse, summary="修改风险规则")
def update_risk_rule(rule_id: int, rule_update: RiskRuleUpdate, db: Session = Depends(get_db)):
    db_rule = db.query(RiskRule).filter(RiskRule.rule_id == rule_id).first()
    if not db_rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    update_data = rule_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_rule, key, value)
    
    db.commit()
    db.refresh(db_rule)
    return db_rule

@router.delete("/rules/{rule_id}", summary="删除风险规则")
def delete_risk_rule(rule_id: int, db: Session = Depends(get_db)):
    db_rule = db.query(RiskRule).filter(RiskRule.rule_id == rule_id).first()
    if not db_rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    
    db.delete(db_rule)
    db.commit()
    return {"message": "规则已删除"}

# =======================
# 2. 号码黑名单管理 (Blacklist)
# =======================

@router.get("/blacklist", response_model=List[BlacklistResponse], summary="获取黑名单列表")
def get_blacklist(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    blacklist = db.query(NumberBlacklist).offset(skip).limit(limit).all()
    return blacklist

@router.post("/blacklist", response_model=BlacklistResponse, summary="封禁号码（添加黑名单）")
def add_to_blacklist(item: BlacklistCreate, db: Session = Depends(get_db)):
    existing = db.query(NumberBlacklist).filter(NumberBlacklist.number == item.number).first()
    if existing:
        raise HTTPException(status_code=400, detail="该号码已在黑名单中")
    
    db_item = NumberBlacklist(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

@router.delete("/blacklist/{id}", summary="移除黑名单")
def remove_from_blacklist(id: int, db: Session = Depends(get_db)):
    db_item = db.query(NumberBlacklist).filter(NumberBlacklist.id == id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="记录不存在")
    
    db.delete(db_item)
    db.commit()
    return {"message": "号码已移除黑名单"}