"""
导入案例库和法律库到向量数据库
用于用户学习推荐系统
"""
import os
import sys
import json
import asyncio
from typing import List, Dict
from datetime import datetime
from unittest.mock import MagicMock

# 禁用ChromaDB遥测
os.environ['ANONYMIZED_TELEMETRY'] = 'false'
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['POSTHOG_DISABLED'] = '1'

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock掉不需要的模块
sys.modules['chromadb.telemetry.posthog'] = MagicMock()
sys.modules['posthog'] = MagicMock()
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.websockets'] = MagicMock()
sys.modules['aiosmtplib'] = MagicMock()

from app.services.vector_db_service import vector_db
from app.core.logger import get_logger

logger = get_logger(__name__)


# 诈骗类型映射（统一格式）
FRAUD_TYPE_MAPPING = {
    # 常见诈骗类型
    "冒充公检法": "impersonate_police",
    "冒充公检法诈骗": "impersonate_police",
    "虚假贷款": "fake_loan",
    "虚假贷款诈骗": "fake_loan",
    "刷单返利": "brush_order",
    "刷单返利诈骗": "brush_order",
    "虚假投资理财": "fake_investment",
    "虚假投资理财诈骗": "fake_investment",
    "冒充客服": "impersonate_customer_service",
    "冒充客服诈骗": "impersonate_customer_service",
    "游戏产品虚假交易": "game_transaction",
    "游戏交易": "game_transaction",
    "杀猪盘": "romance_scam",
    "网络交友": "romance_scam",
    "冒充领导": "impersonate_leader",
    "冒充熟人": "impersonate_acquaintance",
    "快递理赔": "express_compensation",
    "虚假征信": "fake_credit",
    "裸聊敲诈": "sextortion",
    "网络博彩": "online_gambling",
}

# 目标人群映射
TARGET_GROUP_MAPPING = {
    "游戏玩家": "game_player",
    "学生": "student",
    "大学生": "student",
    "职场人群": "worker",
    "上班族": "worker",
    "投资者": "investor",
    "借贷人群": "borrower",
    "单身人群": "single",
    "老年人": "elderly",
    "老人": "elderly",
    "经常网购者": "online_shopper",
    "宝妈": "mother",
}


def normalize_fraud_type(fraud_type: str) -> str:
    """标准化诈骗类型"""
    if not fraud_type:
        return "other"
    fraud_type = fraud_type.strip()
    return FRAUD_TYPE_MAPPING.get(fraud_type, fraud_type.lower().replace(" ", "_"))


def normalize_target_groups(target_group_str: str) -> List[str]:
    """标准化目标人群"""
    if not target_group_str:
        return []
    groups = [g.strip() for g in target_group_str.split(",")]
    return [TARGET_GROUP_MAPPING.get(g, g) for g in groups if g]


def parse_risk_level(risk_level: str) -> int:
    """解析风险等级为数字"""
    if not risk_level:
        return 1
    risk_level = risk_level.strip()
    if "极高" in risk_level or "最高" in risk_level:
        return 5
    elif "高" in risk_level:
        return 4
    elif "中" in risk_level:
        return 3
    elif "低" in risk_level:
        return 2
    return 1


async def import_case_library(file_path: str):
    """导入案例库"""
    logger.info(f"开始导入案例库: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    imported_count = 0
    
    for case in cases:
        try:
            # 提取字段
            title = case.get("title", "")
            content = case.get("content", "")
            fraud_type = normalize_fraud_type(case.get("fraud_type", ""))
            target_groups = normalize_target_groups(case.get("target_group", ""))
            risk_level = parse_risk_level(case.get("risk_level", ""))
            
            if not content:
                continue
            
            # 构建元数据
            metadata = {
                "title": title,
                "fraud_type": fraud_type,
                "target_groups": target_groups,
                "risk_level": risk_level,
                "source": "case_library",
                "imported_at": datetime.now().isoformat()
            }
            
            # 添加到向量数据库
            await vector_db.add_case(
                case_id=f"case_{imported_count}_{datetime.now().timestamp()}",
                content=content,
                fraud_type=fraud_type,
                metadata=metadata
            )
            
            imported_count += 1
            
            if imported_count % 10 == 0:
                logger.info(f"已导入 {imported_count} 个案例")
                
        except Exception as e:
            logger.error(f"导入案例失败: {e}")
            continue
    
    logger.info(f"案例库导入完成，共导入 {imported_count} 个案例")
    return imported_count


async def import_law_library(file_path: str):
    """导入法律库"""
    logger.info(f"开始导入法律库: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        laws = json.load(f)
    
    imported_count = 0
    
    for law in laws:
        try:
            # 提取字段
            title = law.get("title", "")
            content = law.get("content", "")
            fraud_type = normalize_fraud_type(law.get("fraud_type", ""))
            
            if not content:
                continue
            
            # 构建元数据
            metadata = {
                "title": title,
                "fraud_type": fraud_type,
                "source": "law_library",
                "law_type": "反电信网络诈骗法",
                "imported_at": datetime.now().isoformat()
            }
            
            # 添加到向量数据库（作为案例存储，但标记为法律条文）
            await vector_db.add_case(
                case_id=f"law_{imported_count}_{datetime.now().timestamp()}",
                content=f"{title}\n\n{content}",
                fraud_type=fraud_type,
                metadata=metadata
            )
            
            imported_count += 1
            
        except Exception as e:
            logger.error(f"导入法律条文失败: {e}")
            continue
    
    logger.info(f"法律库导入完成，共导入 {imported_count} 条法律")
    return imported_count


async def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases_dir = os.path.join(base_dir, "data", "competition_cases")
    
    # 导入案例库
    case_files = [
        "300数据集标注.json",
        "300数据集标注 (2).json"
    ]
    
    total_cases = 0
    for case_file in case_files:
        file_path = os.path.join(cases_dir, case_file)
        if os.path.exists(file_path):
            count = await import_case_library(file_path)
            total_cases += count
        else:
            logger.warning(f"案例文件不存在: {file_path}")
    
    # 导入法律库
    law_files = [
        "法律234.json",
        "法律567.json"
    ]
    
    total_laws = 0
    for law_file in law_files:
        file_path = os.path.join(cases_dir, law_file)
        if os.path.exists(file_path):
            count = await import_law_library(file_path)
            total_laws += count
        else:
            logger.warning(f"法律文件不存在: {file_path}")
    
    logger.info("=" * 60)
    logger.info("导入完成统计:")
    logger.info(f"  案例库: {total_cases} 个")
    logger.info(f"  法律库: {total_laws} 条")
    logger.info(f"  总计: {total_cases + total_laws} 条")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
