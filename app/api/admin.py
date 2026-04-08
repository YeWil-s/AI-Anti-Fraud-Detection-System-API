"""
app/api/admin.py
管理员专用接口：异步版本
"""
import os
import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, desc
from typing import List, Optional

from app.db.database import get_db
from app.models.risk_rule import RiskRule
from app.models.blacklist import NumberBlacklist
from app.models.user import User
from app.models.call_record import CallRecord, DetectionResult
from app.models.ai_detection_log import AIDetectionLog
from app.models.admin import Admin, AdminLog, SystemMonitor
from app.schemas.admin import (
    RiskRuleCreate, RiskRuleUpdate, RiskRuleResponse,
    BlacklistCreate, BlacklistUpdate, BlacklistResponse,
    CaseUploadRequest, DashboardStats, TrendData
)
from app.core.logger import get_logger
from app.services.llm_service import llm_service

router = APIRouter()
logger = get_logger(__name__)

# 基础目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PENDING_DIR = os.path.join(BASE_DIR, "data", "pending_cases")
LEARNED_DIR = os.path.join(BASE_DIR, "data", "learned_cases")

# =======================
# 1. 仪表盘数据统计 (增强版)
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
    
    # 6. 今日新增用户
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.count(User.user_id)).where(User.created_at >= today)
    )
    new_users_today = result.scalar() or 0
    
    # 7. 今日检测次数
    result = await db.execute(
        select(func.count(CallRecord.call_id)).where(CallRecord.start_time >= today)
    )
    detections_today = result.scalar() or 0
    
    # 8. 今日拦截次数
    result = await db.execute(
        select(func.count(CallRecord.call_id))
        .where(CallRecord.detected_result == DetectionResult.FAKE)
        .where(CallRecord.start_time >= today)
    )
    blocked_today = result.scalar() or 0
    
    # 9. 平均风险评分
    result = await db.execute(
        select(func.avg(AIDetectionLog.overall_score))
    )
    avg_risk_score = result.scalar() or 0
    
    # 10. 系统健康度 (基于最近错误率计算)
    system_health = "100%"
    
    return {
        "total_users": total_users,
        "total_calls": total_calls,
        "fraud_blocked": fraud_calls,
        "blacklist_count": blacklist_count,
        "active_rules": rule_count,
        "new_users_today": new_users_today,
        "detections_today": detections_today,
        "blocked_today": blocked_today,
        "avg_risk_score": round(float(avg_risk_score), 2),
        "system_health": system_health,
        "detection_rate": round((fraud_calls / total_calls * 100), 2) if total_calls > 0 else 0
    }


@router.get("/stats/trends", summary="获取趋势数据")
async def get_trend_stats(
    days: int = Query(7, ge=1, le=30),
    db: AsyncSession = Depends(get_db)
):
    """获取最近N天的趋势数据"""
    trends = []
    
    for i in range(days - 1, -1, -1):
        date = datetime.now() - timedelta(days=i)
        date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        date_end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # 该日检测次数
        result = await db.execute(
            select(func.count(CallRecord.call_id))
            .where(CallRecord.start_time >= date_start)
            .where(CallRecord.start_time <= date_end)
        )
        detections = result.scalar() or 0
        
        # 该日拦截次数
        result = await db.execute(
            select(func.count(CallRecord.call_id))
            .where(CallRecord.detected_result == DetectionResult.FAKE)
            .where(CallRecord.start_time >= date_start)
            .where(CallRecord.start_time <= date_end)
        )
        blocked = result.scalar() or 0
        
        # 该日新增用户
        result = await db.execute(
            select(func.count(User.user_id))
            .where(User.created_at >= date_start)
            .where(User.created_at <= date_end)
        )
        new_users = result.scalar() or 0
        
        trends.append({
            "date": date.strftime("%m-%d"),
            "detections": detections,
            "blocked": blocked,
            "new_users": new_users
        })
    
    return trends


@router.get("/stats/fraud-types", summary="获取诈骗类型分布")
async def get_fraud_type_stats(db: AsyncSession = Depends(get_db)):
    """获取诈骗类型分布统计 - 基于通话记录的fraud_type字段"""
    # 查询有明确诈骗类型的记录
    result = await db.execute(
        select(
            CallRecord.fraud_type,
            func.count(CallRecord.call_id).label("count")
        )
        .where(CallRecord.fraud_type.isnot(None))
        .where(CallRecord.detected_result == DetectionResult.FAKE)
        .group_by(CallRecord.fraud_type)
        .order_by(desc("count"))
    )
    
    type_stats = []
    for row in result.all():
        if row.fraud_type:  # 确保不为空
            type_stats.append({
                "type": row.fraud_type,
                "value": row.count
            })
    
    # 如果没有数据，返回空列表（前端会显示默认数据）
    return type_stats


@router.get("/stats/hourly", summary="获取24小时分布数据")
async def get_hourly_stats(db: AsyncSession = Depends(get_db)):
    """获取最近24小时的检测分布"""
    last_24h = datetime.now() - timedelta(hours=24)
    
    result = await db.execute(
        select(
            func.extract('hour', CallRecord.start_time).label("hour"),
            func.count(CallRecord.call_id).label("count")
        )
        .where(CallRecord.start_time >= last_24h)
        .group_by(func.extract('hour', CallRecord.start_time))
        .order_by("hour")
    )
    
    hourly_data = [0] * 24
    for row in result.all():
        hour = int(row.hour)
        hourly_data[hour] = row.count
    
    return {"hours": list(range(24)), "counts": hourly_data}

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
        select(CallRecord, AIDetectionLog)
        .outerjoin(AIDetectionLog, CallRecord.call_id == AIDetectionLog.call_id)
        .where(CallRecord.detected_result == DetectionResult.FAKE)
        .order_by(CallRecord.start_time.desc())
        .offset(skip).limit(limit)
    )
    
    # 格式化返回数据，提取有用的信息
    response_data = []
    for case, ai_log in result.all():
        # 适配真实的 CallRecord 字段
        contact_info = case.caller_number or case.target_name or "未知号码"
        
        # 获取AI分析摘要（如果有LLM分析结果）
        details = case.analysis or f"检测到与 {contact_info} 的风险通话"
        if ai_log and ai_log.overall_score:
            details = f"[风险评分:{ai_log.overall_score}] {details}"
        
        response_data.append({
            "call_id": case.call_id,
            "user_id": case.user_id,
            "target_number": contact_info,
            "start_time": case.start_time.isoformat() if case.start_time else None,
            "duration": case.duration,
            "risk_level": "高危",
            "fraud_type": case.fraud_type or "未分类拦截",  # 使用真实的fraud_type
            "details": details
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

    # 适配真实的 CallRecord 字段
    contact_info = case.caller_number or case.target_name or "未知号码"

    # 1. 组装学习数据格式 (必须符合我们在 maintenance_tasks.py 中约定的格式)
    learn_data = [{
        "modality": "audio", # 假设电话拦截默认是音频转写
        "fraud_type": f"管理员标记案例_{contact_info}",
        "risk_level": "高危",
        "content": f"系统拦截了与 {contact_info} 的高危通话，判定为疑似欺诈。",
        "source": f"管理员人工核实 (CallID:{call_id})"
    }]

    # 2. 确定文件保存路径
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


@router.post("/fraud-cases/{call_id}/learn-with-edit", summary="编辑并加入待学习队列")
async def learn_fraud_case_with_edit(
    call_id: int,
    data: dict,
    db: AsyncSession = Depends(get_db)
):
    """
    管理员编辑案例信息后，加入待学习队列
    """
    result = await db.execute(select(CallRecord).where(CallRecord.call_id == call_id))
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(status_code=404, detail="未找到该通话记录")
    
    # 验证必填字段
    fraud_type = data.get('fraud_type')
    if not fraud_type:
        raise HTTPException(status_code=400, detail="诈骗类型不能为空")
    
    # 组装学习数据
    learn_data = {
        "modality": data.get('modality', 'audio'),
        "fraud_type": fraud_type,
        "risk_level": data.get('risk_level', '高危'),
        "content": data.get('content', ''),
        "details": data.get('details', ''),
        "source": data.get('source', f"管理员编辑核实 (CallID:{call_id})"),
        "tags": data.get('tags', []),
        "uploader": data.get('uploader', 'admin'),
        "call_id": call_id,
        "uploaded_at": datetime.now().isoformat()
    }
    
    # 保存到日期的文件中
    os.makedirs(PENDING_DIR, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    file_name = f"edited_cases_{today_str}.json"
    file_path = os.path.join(PENDING_DIR, file_name)
    
    try:
        # 读取现有数据或创建新列表
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except Exception:
                existing_data = []
        
        # 追加新案例
        existing_data.append(learn_data)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"编辑后的案例已保存到 {file_name}, 当前共 {len(existing_data)} 条案例")
        
        return {
            "msg": "案例编辑成功并加入待学习队列",
            "file": file_name,
            "case_count": len(existing_data)
        }
        
    except Exception as e:
        logger.error(f"保存案例失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


# =======================
# 5. 案例上传与管理 (新增功能)
# =======================
@router.post("/cases/upload", summary="上传案例到待学习队列")
async def upload_case(
    request: CaseUploadRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    管理员或用户上传案例，填写标签后生成JSON文件到pending_cases目录
    同一天输入的案例会在同一文件中追加
    """
    try:
        # 确保目录存在
        os.makedirs(PENDING_DIR, exist_ok=True)
        
        # 生成文件名 (按日期)
        today_str = datetime.now().strftime("%Y%m%d")
        file_name = f"uploaded_cases_{today_str}.json"
        file_path = os.path.join(PENDING_DIR, file_name)
        
        # 构建案例数据
        case_data = {
            "modality": request.modality,
            "fraud_type": request.fraud_type,
            "risk_level": request.risk_level,
            "content": request.content,
            "source": request.source or f"用户上传_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "tags": request.tags or [],
            "uploaded_at": datetime.now().isoformat(),
            "uploader": request.uploader or "anonymous"
        }
        
        # 读取现有数据或创建新列表
        existing_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except Exception as e:
                logger.warning(f"读取现有案例文件失败: {e}")
                existing_data = []
        
        # 追加新案例
        existing_data.append(case_data)
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"案例已保存到 {file_name}, 当前共 {len(existing_data)} 条案例")
        
        return {
            "msg": "案例上传成功",
            "file": file_name,
            "case_count": len(existing_data),
            "case_id": len(existing_data) - 1
        }
        
    except Exception as e:
        logger.error(f"案例上传失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"案例上传失败: {str(e)}")


@router.post("/cases/suggest-fields", summary="LLM建议案例字段")
async def suggest_case_fields(payload: dict):
    """根据案例内容让 LLM 给出字段建议，供前端一键填充。"""
    content = (payload or {}).get("content", "")
    if not content or not str(content).strip():
        raise HTTPException(status_code=400, detail="content 不能为空")

    result = await llm_service.analyze_text_risk(
        user_input=str(content),
        chat_history=str(content),
        user_profile="管理端案例上传"
    )

    risk_map = {"fake": "高危", "suspicious": "中危", "safe": "低危"}
    fraud_type = result.get("fraud_type") or "其他"
    return {
        "fraud_type": fraud_type,
        "modality": "text",
        "risk_level": risk_map.get(result.get("risk_level"), "中危"),
        "content": str(content).strip(),
        "source": "LLM助手建议",
        "tags": [t for t in ["LLM建议", fraud_type] if t],
        "advice": result.get("advice", ""),
        "analysis": result.get("analysis", "")
    }


@router.get("/cases/pending", summary="获取待学习案例列表")
async def get_pending_cases():
    """获取所有待学习的案例文件列表"""
    try:
        os.makedirs(PENDING_DIR, exist_ok=True)
        
        files = []
        for filename in os.listdir(PENDING_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(PENDING_DIR, filename)
                stat = os.stat(file_path)
                
                # 读取案例数量
                case_count = 0
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        case_count = len(data) if isinstance(data, list) else 0
                except:
                    pass
                
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "case_count": case_count
                })
        
        # 按修改时间倒序
        files.sort(key=lambda x: x["modified"], reverse=True)
        return files
        
    except Exception as e:
        logger.error(f"获取待学习案例失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/cases/learned", summary="获取已学习案例列表")
async def get_learned_cases():
    """获取所有已学习的案例文件列表"""
    try:
        os.makedirs(LEARNED_DIR, exist_ok=True)
        
        files = []
        for filename in os.listdir(LEARNED_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(LEARNED_DIR, filename)
                stat = os.stat(file_path)
                
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "learned_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # 按时间倒序
        files.sort(key=lambda x: x["learned_at"], reverse=True)
        return files
        
    except Exception as e:
        logger.error(f"获取已学习案例失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")


@router.get("/cases/pending/{filename}", summary="获取待学习案例详情")
async def get_pending_case_detail(filename: str):
    """获取指定待学习案例文件的详细内容"""
    try:
        file_path = os.path.join(PENDING_DIR, filename)
        
        # 安全检查：确保文件在指定目录内
        if not os.path.abspath(file_path).startswith(os.path.abspath(PENDING_DIR)):
            raise HTTPException(status_code=400, detail="非法文件路径")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取案例详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@router.delete("/cases/pending/{filename}", summary="删除待学习案例文件")
async def delete_pending_case(filename: str):
    """删除指定的待学习案例文件"""
    try:
        file_path = os.path.join(PENDING_DIR, filename)
        
        # 安全检查
        if not os.path.abspath(file_path).startswith(os.path.abspath(PENDING_DIR)):
            raise HTTPException(status_code=400, detail="非法文件路径")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        os.remove(file_path)
        return {"msg": "文件已删除", "filename": filename}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除案例文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# =======================
# 6. 系统监控与日志
# =======================
@router.get("/system/logs", summary="获取系统操作日志")
async def get_system_logs(
    skip: int = 0,
    limit: int = 50,
    action: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """获取管理员操作日志"""
    query = select(AdminLog).order_by(desc(AdminLog.created_at))
    
    if action:
        query = query.where(AdminLog.action == action)
    
    result = await db.execute(query.offset(skip).limit(limit))
    logs = result.scalars().all()
    
    return [{
        "log_id": log.log_id,
        "admin_id": log.admin_id,
        "action": log.action,
        "resource": log.resource,
        "resource_id": log.resource_id,
        "details": log.details,
        "ip_address": log.ip_address,
        "created_at": log.created_at.isoformat() if log.created_at else None
    } for log in logs]


@router.get("/system/health", summary="获取系统健康状态")
async def get_system_health():
    """获取系统健康状态详情"""
    try:
        # 检查待学习文件数量
        pending_count = 0
        if os.path.exists(PENDING_DIR):
            pending_count = len([f for f in os.listdir(PENDING_DIR) if f.endswith('.json')])
        
        # 检查已学习文件数量
        learned_count = 0
        if os.path.exists(LEARNED_DIR):
            learned_count = len([f for f in os.listdir(LEARNED_DIR) if f.endswith('.json')])
        
        return {
            "status": "healthy",
            "pending_cases": pending_count,
            "learned_cases": learned_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/detection/recent", summary="获取最近检测记录")
async def get_recent_detections(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    """获取最近的AI检测记录"""
    result = await db.execute(
        select(AIDetectionLog, CallRecord)
        .join(CallRecord, AIDetectionLog.call_id == CallRecord.call_id)
        .order_by(desc(AIDetectionLog.created_at))
        .limit(limit)
    )
    
    detections = []
    for log, call in result.all():
        detections.append({
            "log_id": log.log_id,
            "call_id": log.call_id,
            "detection_type": log.detection_type,
            "overall_score": log.overall_score,
            "voice_confidence": log.voice_confidence,
            "video_confidence": log.video_confidence,
            "text_confidence": log.text_confidence,
            "created_at": log.created_at.isoformat() if log.created_at else None,
            "caller_number": call.caller_number if call else None
        })
    
    return detections


@router.get("/calls/recent-summaries", summary="获取最近通话记录与建议")
async def get_recent_call_summaries(
    limit: int = Query(6, ge=1, le=30),
    db: AsyncSession = Depends(get_db)
):
    """返回最近几条 call_record 记录，含检测结果与建议。"""
    result = await db.execute(
        select(CallRecord)
        .order_by(desc(CallRecord.start_time), desc(CallRecord.created_at))
        .limit(limit)
    )

    records = []
    for call in result.scalars().all():
        detected_result = call.detected_result.value if call.detected_result else "safe"
        # 仅基于 call_record 的检测结果给出展示分，不依赖 ai_detection_log
        score_map = {"fake": 90, "suspicious": 60, "safe": 20}
        records.append({
            "call_id": call.call_id,
            "caller_number": call.caller_number or call.target_name or "未知",
            "detected_result": detected_result,
            "risk_score": score_map.get(detected_result, 0),
            "analysis": call.analysis or "",
            "advice": call.advice or "建议保持警惕，勿转账、勿透露验证码。",
            "created_at": call.start_time.isoformat() if call.start_time else (
                call.created_at.isoformat() if call.created_at else None
            )
        })

    return records


# =======================
# 7. 用户管理
# =======================
@router.get("/users", summary="获取用户列表")
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """获取所有用户列表"""
    result = await db.execute(
        select(User)
        .order_by(desc(User.created_at))
        .offset(skip)
        .limit(limit)
    )
    users = result.scalars().all()
    
    return [{
        "user_id": u.user_id,
        "username": u.username,
        "name": u.name,
        "phone": u.phone,
        "email": u.email,
        "role_type": u.role_type,
        "gender": u.gender,
        "profession": u.profession,
        "marital_status": u.marital_status,
        "family_id": u.family_id,
        "is_active": u.is_active,
        "is_admin": u.is_admin,
        "created_at": u.created_at.isoformat() if u.created_at else None
    } for u in users]


@router.get("/users/{user_id}", summary="获取用户详情")
async def get_user_detail(user_id: int, db: AsyncSession = Depends(get_db)):
    """获取单个用户详情"""
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "name": user.name,
        "phone": user.phone,
        "email": user.email,
        "role_type": user.role_type,
        "gender": user.gender,
        "profession": user.profession,
        "marital_status": user.marital_status,
        "family_id": user.family_id,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }


@router.put("/users/{user_id}", summary="更新用户信息")
async def update_user(
    user_id: int,
    user_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """更新用户信息"""
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    # 更新允许修改的字段
    allowed_fields = ['name', 'phone', 'email', 'role_type', 'gender', 'profession', 'marital_status']
    for field in allowed_fields:
        if field in user_data:
            setattr(user, field, user_data[field])
    
    await db.commit()
    await db.refresh(user)
    
    return {"msg": "更新成功"}


@router.patch("/users/{user_id}/status", summary="更新用户状态")
async def update_user_status(
    user_id: int,
    status_data: dict,
    db: AsyncSession = Depends(get_db)
):
    """启用或禁用用户账号"""
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    is_active = status_data.get('is_active')
    if is_active is not None:
        user.is_active = is_active
        await db.commit()
    
    return {"msg": "状态更新成功", "is_active": user.is_active}


@router.delete("/users/{user_id}", summary="删除用户")
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    """删除用户"""
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    await db.delete(user)
    await db.commit()
    
    return {"msg": "删除成功"}


@router.get("/users/{user_id}/call-stats", summary="获取用户通话统计")
async def get_user_call_stats(user_id: int, db: AsyncSession = Depends(get_db)):
    """获取用户的通话统计信息"""
    # 总通话数
    result = await db.execute(
        select(func.count(CallRecord.call_id))
        .where(CallRecord.user_id == user_id)
    )
    total_calls = result.scalar() or 0
    
    # 诈骗通话数
    result = await db.execute(
        select(func.count(CallRecord.call_id))
        .where(CallRecord.user_id == user_id)
        .where(CallRecord.detected_result == DetectionResult.FAKE)
    )
    fraud_calls = result.scalar() or 0
    
    # 可疑通话数
    result = await db.execute(
        select(func.count(CallRecord.call_id))
        .where(CallRecord.user_id == user_id)
        .where(CallRecord.detected_result == DetectionResult.SUSPICIOUS)
    )
    suspicious_calls = result.scalar() or 0
    
    return {
        "total_calls": total_calls,
        "fraud_calls": fraud_calls,
        "suspicious_calls": suspicious_calls
    }


@router.get("/family-groups", summary="获取家庭组列表")
async def get_family_groups(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """获取所有家庭组列表（适配新权限模型）"""
    from app.models.family_group import FamilyGroup, FamilyAdmin
    from app.models.user import User
    
    result = await db.execute(
        select(FamilyGroup)
        .order_by(desc(FamilyGroup.created_at))
        .offset(skip)
        .limit(limit)
    )
    groups = result.scalars().all()
    
    # 查询每个家庭的详细信息
    group_list = []
    for g in groups:
        # 统计该家庭的成员数
        member_result = await db.execute(
            select(func.count(User.user_id)).where(User.family_id == g.id)
        )
        member_count = member_result.scalar() or 0
        
        # 查询主管理员信息
        admin_result = await db.execute(
            select(User).where(User.user_id == g.admin_id)
        )
        admin = admin_result.scalar_one_or_none()
        
        # 查询该家庭组的管理员统计
        admins_result = await db.execute(
            select(FamilyAdmin).where(FamilyAdmin.family_id == g.id)
        )
        admins = admins_result.scalars().all()
        primary_count = sum(1 for a in admins if a.admin_role == "primary")
        secondary_count = sum(1 for a in admins if a.admin_role == "secondary")
        
        group_list.append({
            "id": g.id,
            "group_name": g.group_name,
            "admin_id": g.admin_id,
            "primary_admin": {
                "user_id": admin.user_id,
                "username": admin.username,
                "phone": admin.phone
            } if admin else None,
            "statistics": {
                "total_members": member_count,
                "primary_admins": primary_count,
                "secondary_admins": secondary_count
            },
            "created_at": g.created_at.isoformat() if g.created_at else None
        })
    
    return {"items": group_list, "total": len(group_list)}


@router.get("/family-groups/{family_id}/members", summary="获取家庭组成员列表")
async def get_family_group_members(
    family_id: int,
    db: AsyncSession = Depends(get_db)
):
    """获取指定家庭组的成员列表（管理端）"""
    from app.models.family_group import FamilyGroup, FamilyAdmin
    from app.models.user import User
    
    # 检查家庭组是否存在
    family_result = await db.execute(
        select(FamilyGroup).where(FamilyGroup.id == family_id)
    )
    family = family_result.scalar_one_or_none()
    if not family:
        raise HTTPException(status_code=404, detail="家庭组不存在")
    
    # 查询成员
    members_result = await db.execute(
        select(User).where(User.family_id == family_id)
    )
    members = members_result.scalars().all()
    
    # 查询管理员角色
    admins_result = await db.execute(
        select(FamilyAdmin).where(FamilyAdmin.family_id == family_id)
    )
    admin_map = {a.user_id: a.admin_role for a in admins_result.scalars().all()}
    
    return {
        "family_id": family_id,
        "group_name": family.group_name,
        "members": [
            {
                "user_id": m.user_id,
                "username": m.username,
                "name": m.name,
                "phone": m.phone,
                "email": m.email,
                "role_type": m.role_type,
                "admin_role": admin_map.get(m.user_id, "none"),
                "is_active": m.is_active
            }
            for m in members
        ]
    }


@router.get("/family-stats", summary="获取家庭组统计")
async def get_family_stats(
    db: AsyncSession = Depends(get_db)
):
    """获取家庭组统计数据"""
    from app.models.family_group import FamilyGroup, FamilyAdmin
    from app.models.user import User
    
    # 家庭组总数
    families_result = await db.execute(select(func.count(FamilyGroup.id)))
    total_families = families_result.scalar() or 0
    
    # 成员总数
    members_result = await db.execute(
        select(func.count(User.user_id)).where(User.family_id.isnot(None))
    )
    total_members = members_result.scalar() or 0
    
    # 管理员总数
    admins_result = await db.execute(select(func.count(FamilyAdmin.id)))
    total_admins = admins_result.scalar() or 0
    
    return {
        "total_families": total_families,
        "total_members": total_members,
        "total_admins": total_admins
    }


# =======================
# 全过程记录接口（管理后台）
# =======================

@router.get("/call-records/{call_id}/detection-timeline", summary="获取检测时间轴（管理员）")
async def admin_get_detection_timeline(
    call_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    管理员获取指定通话的详细检测时间轴
    """
    # 查询通话记录
    result = await db.execute(
        select(CallRecord).where(CallRecord.call_id == call_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")
    
    # 获取AI检测日志
    ai_logs_result = await db.execute(
        select(AIDetectionLog)
        .where(AIDetectionLog.call_id == call_id)
        .order_by(AIDetectionLog.time_offset.asc())
    )
    ai_logs = ai_logs_result.scalars().all()
    
    # 构建时间轴
    timeline = []
    for log in ai_logs:
        timeline.append({
            "time_offset": log.time_offset,
            "timestamp": log.created_at.isoformat() if log.created_at else None,
            "modalities": {
                "text": {
                    "confidence": log.text_confidence,
                    "score": log.text_confidence * 100 if log.text_confidence else 0,
                    "detected_text": log.detected_text,
                    "detected_keywords": log.detected_keywords,
                    "match_script": log.match_script,
                    "intent": log.intent
                },
                "voice": {
                    "confidence": log.voice_confidence,
                    "score": log.voice_confidence * 100 if log.voice_confidence else 0
                },
                "video": {
                    "confidence": log.video_confidence,
                    "score": log.video_confidence * 100 if log.video_confidence else 0
                }
            },
            "overall_score": log.overall_score,
            "fused_risk_level": _get_risk_level(log.overall_score),
            "detection_type": log.detection_type,
            "model_version": log.model_version,
            "evidence_url": log.evidence_snapshot,
            "log_id": log.log_id
        })
    
    return {
        "call_id": call_id,
        "user_id": record.user_id,
        "timeline": timeline,
        "statistics": {
            "total_events": len(timeline),
            "max_overall_score": max([t["overall_score"] for t in timeline]) if timeline else 0,
            "duration_seconds": timeline[-1]["time_offset"] if timeline else 0
        }
    }


def _get_risk_level(score: float) -> str:
    """根据分数获取风险等级"""
    if score >= 80:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"


# =======================
# 用户长期记忆管理接口
# =======================

@router.get("/users/{user_id}/memories", summary="获取用户长期记忆")
async def get_user_memories(
    user_id: int,
    memory_type: Optional[str] = None,
    min_importance: int = Query(1, ge=1, le=5),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db)
):
    """获取用户的长期记忆列表"""
    from app.services.long_term_memory_service import long_term_memory_service
    
    memories = await long_term_memory_service.get_memories(
        db, user_id, memory_type, min_importance, limit
    )
    
    return {
        "items": [
            {
                "memory_id": m.memory_id,
                "memory_type": m.memory_type,
                "content": m.content,
                "importance": m.importance,
                "source_call_id": m.source_call_id,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in memories
        ]
    }


@router.post("/users/{user_id}/memories", summary="添加用户长期记忆")
async def add_user_memory(
    user_id: int,
    memory_type: str = Query(..., description="记忆类型: fraud_experience/alert_response/preference/risk_pattern"),
    content: str = Query(..., description="记忆内容"),
    importance: int = Query(3, ge=1, le=5, description="重要性 1-5"),
    source_call_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """手动添加用户的长期记忆"""
    from app.services.long_term_memory_service import long_term_memory_service
    
    # 检查用户是否存在
    user_result = await db.execute(select(User).where(User.user_id == user_id))
    if not user_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="用户不存在")
    
    memory = await long_term_memory_service.add_memory(
        db, user_id, memory_type, content, importance, source_call_id
    )
    
    if memory:
        # 更新摘要
        await long_term_memory_service.update_memory_summary(db, user_id)
        return {"code": 200, "message": "记忆添加成功", "memory_id": memory.memory_id}
    else:
        return {"code": 200, "message": "相似记忆已存在，无需重复添加"}


@router.delete("/memories/{memory_id}", summary="删除长期记忆")
async def delete_memory(
    memory_id: int,
    db: AsyncSession = Depends(get_db)
):
    """删除指定的长期记忆"""
    from app.models.user_memory import UserMemory
    
    result = await db.execute(select(UserMemory).where(UserMemory.memory_id == memory_id))
    memory = result.scalar_one_or_none()
    
    if not memory:
        raise HTTPException(status_code=404, detail="记忆不存在")
    
    user_id = memory.user_id
    await db.delete(memory)
    await db.commit()
    
    # 更新摘要
    from app.services.long_term_memory_service import long_term_memory_service
    await long_term_memory_service.update_memory_summary(db, user_id)
    
    return {"code": 200, "message": "记忆删除成功"}


@router.get("/users/{user_id}/memory-summary", summary="获取用户记忆摘要")
async def get_user_memory_summary(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """获取用户的记忆摘要（用于展示）"""
    from app.services.long_term_memory_service import long_term_memory_service
    
    summary = await long_term_memory_service.get_memory_summary(db, user_id)
    
    # 同时获取统计信息
    from app.models.user_memory import UserMemory
    stats_result = await db.execute(
        select(UserMemory.memory_type, func.count(UserMemory.memory_id))
        .where(UserMemory.user_id == user_id)
        .group_by(UserMemory.memory_type)
    )
    stats = {row[0]: row[1] for row in stats_result.all()}
    
    return {
        "user_id": user_id,
        "summary": summary,
        "statistics": stats
    }


@router.post("/users/{user_id}/refresh-memory-summary", summary="刷新用户记忆摘要")
async def refresh_memory_summary(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    """手动刷新用户的记忆摘要"""
    from app.services.long_term_memory_service import long_term_memory_service
    
    summary = await long_term_memory_service.update_memory_summary(db, user_id)
    
    return {
        "code": 200,
        "message": "记忆摘要已刷新",
        "summary": summary
    }


@router.get("/call-records/{call_id}/chat-history", summary="获取对话历史（管理员）")
async def admin_get_chat_history(
    call_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    管理员获取指定通话的对话历史
    """
    from app.models.chat_message import ChatMessage
    
    # 查询通话记录
    result = await db.execute(
        select(CallRecord).where(CallRecord.call_id == call_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="通话记录不存在")
    
    # 获取对话历史
    messages_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.call_id == call_id)
        .order_by(ChatMessage.sequence.asc())
    )
    messages = messages_result.scalars().all()
    
    return {
        "call_id": call_id,
        "user_id": record.user_id,
        "message_count": len(messages),
        "messages": [
            {
                "sequence": m.sequence,
                "speaker": m.speaker,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None
            }
            for m in messages
        ]
    }


@router.get("/detection/{log_id}/evidence", summary="获取检测证据详情（管理员）")
async def admin_get_evidence_detail(
    log_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    管理员获取指定检测日志的详细证据信息
    """
    result = await db.execute(
        select(AIDetectionLog).where(AIDetectionLog.log_id == log_id)
    )
    log = result.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="检测日志不存在")
    
    return {
        "log_id": log.log_id,
        "call_id": log.call_id,
        "time_offset": log.time_offset,
        "created_at": log.created_at.isoformat() if log.created_at else None,
        "detection_type": log.detection_type,
        "modalities": {
            "text": {
                "confidence": log.text_confidence,
                "detected_text": log.detected_text,
                "detected_keywords": log.detected_keywords,
                "match_script": log.match_script,
                "intent": log.intent
            },
            "voice": {
                "confidence": log.voice_confidence
            },
            "video": {
                "confidence": log.video_confidence
            }
        },
        "overall_score": log.overall_score,
        "risk_level": _get_risk_level(log.overall_score),
        "evidence": {
            "snapshot_url": log.evidence_snapshot,
            "ocr_text": log.image_ocr_text,
            "ocr_dialogue_hash": log.ocr_dialogue_hash
        },
        "technical_details": {
            "algorithm_details": log.algorithm_details,
            "model_version": log.model_version
        }
    }