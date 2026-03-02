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

@router.get("/fraud-cases", summary="获取被判定为诈骗的通话记录")
async def get_fraud_cases(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    """
    供管理员审核，获取那些 detected_result 为 FAKE 的高危通话
    """
    result = await db.execute(
        select(CallRecord)
        .where(CallRecord.detected_result == DetectionResult.FAKE)
        .order_by(CallRecord.start_time.desc())
        .offset(skip).limit(limit)
    )
    cases = result.scalars().all()
    
    # 格式化返回数据，提取有用的信息
    response_data = []
    for case in cases:
        response_data.append({
            "call_id": case.call_id,
            "user_id": case.user_id,
            "target_number": case.target_number,
            "start_time": case.start_time.isoformat() if case.start_time else None,
            "duration": case.duration,
            "risk_level": "高危",  # 简单映射
            "fraud_type": "未分类拦截", # 真实环境可根据 AI 日志提取，这里暂用默认
            # 如果你有保存最终的话术，可以展示在这里。这里假设你保存了检测详情
            "details": case.detection_details or f"检测到来自 {case.target_number} 的风险通话" 
        })
    return response_data

@router.post("/fraud-cases/{call_id}/learn", summary="将案例加入待学习队列")
async def learn_fraud_case(call_id: int, db: AsyncSession = Depends(get_db)):
    """
    管理员核实后，将此记录转换为 JSON 文件，放入 pending_cases 目录
    供 maintenance_tasks 每天凌晨自动学习
    """
    result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(status_code=404, detail="未找到该通话记录")
    if case.detected_result != DetectionResult.FAKE:
         raise HTTPException(status_code=400, detail="该通话并非诈骗，无需学习")

    # 1. 组装学习数据格式 (必须符合我们在 maintenance_tasks.py 中约定的格式)
    learn_data = [{
        "modality": "audio", # 假设电话拦截默认是音频转写
        "fraud_type": f"管理员标记案例_{case.target_number}",
        "risk_level": "高危",
        "content": case.detection_details or f"系统拦截了与 {case.target_number} 的高危通话，判定为疑似欺诈。",
        "source": f"管理员人工核实 (CallID:{call_id})"
    }]

    # 2. 确定文件保存路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PENDING_DIR = os.path.join(BASE_DIR, "data", "pending_cases")
    os.makedirs(PENDING_DIR, exist_ok=True)
    
    file_name = f"manual_learn_{call_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    file_path = os.path.join(PENDING_DIR, file_name)
    
    # 3. 写入 JSON 文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(learn_data, f, ensure_ascii=False, indent=2)
            
        return {"msg": "成功加入待学习队列", "file": file_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件生成失败: {str(e)}")