from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# --- 风险规则 (Risk Rule) 模型 ---
class RiskRuleBase(BaseModel):
    keyword: str
    risk_level: int = 1
    action: str = "alert"  # alert 或 block
    description: Optional[str] = None
    is_active: bool = True

class RiskRuleCreate(RiskRuleBase):
    pass

class RiskRuleUpdate(BaseModel):
    keyword: Optional[str] = None
    risk_level: Optional[int] = None
    action: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class RiskRuleResponse(RiskRuleBase):
    rule_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True # 适配 SQLAlchemy 对象

# --- 号码黑名单 (Blacklist) 模型 ---
class BlacklistBase(BaseModel):
    number: str
    source: str = "manual_admin" # 默认来源为管理员手动添加
    risk_level: int = 5
    description: Optional[str] = None
    is_active: bool = True

class BlacklistCreate(BlacklistBase):
    pass

class BlacklistUpdate(BaseModel):
    number: Optional[str] = None
    risk_level: Optional[int] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class BlacklistResponse(BlacklistBase):
    id: int
    report_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True