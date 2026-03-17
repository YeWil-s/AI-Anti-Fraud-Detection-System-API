"""
反诈推荐数据导入脚本
将同学收集的案例、标语、视频数据导入 ChromaDB 向量数据库

用法:
    python scripts/import_recommendation_data.py --cases data/cases.json --slogans data/slogans.json --videos data/videos.json
"""
import os
import sys
from unittest.mock import MagicMock

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Mock 掉 telemetry 模块，防止报错
sys.modules['chromadb.telemetry.posthog'] = MagicMock()
sys.modules['posthog'] = MagicMock()

import json
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_db_service import vector_db
from app.core.logger import get_logger

logger = get_logger(__name__)


def import_cases(file_path: str):
    """导入案例数据"""
    if not os.path.exists(file_path):
        logger.warning(f"案例文件不存在: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    
    for i, case in enumerate(cases):
        # 构建文档内容：标题 + 内容
        doc_content = f"{case.get('title', '')}\n{case.get('content', '')}"
        
        documents.append(doc_content)
        metadatas.append({
            "title": case.get('title', ''),
            "fraud_type": case.get('fraud_type', '未知'),
            "risk_level": case.get('risk_level', '未知'),
            "target_group": case.get('target_group', ''),
            "content_type": "case"
        })
        ids.append(f"case_{i}")
    
    # 导入到 ChromaDB
    vector_db.add_data("anti_fraud_cases", documents, metadatas, ids)
    logger.info(f"成功导入 {len(cases)} 条案例数据")


def import_slogans(file_path: str):
    """导入宣传标语数据"""
    if not os.path.exists(file_path):
        logger.warning(f"标语文件不存在: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        slogans = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    
    for i, slogan in enumerate(slogans):
        documents.append(slogan.get('content', ''))
        metadatas.append({
            "content": slogan.get('content', ''),
            "fraud_type": slogan.get('fraud_type', '通用'),
            "content_type": "slogan"
        })
        ids.append(f"slogan_{i}")
    
    # 导入到 ChromaDB - slogans 集合
    vector_db.slogans_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(f"成功导入 {len(slogans)} 条标语数据")


def import_videos(file_path: str):
    """导入宣传视频数据"""
    if not os.path.exists(file_path):
        logger.warning(f"视频文件不存在: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    
    for i, video in enumerate(videos):
        # 构建文档内容：标题 + 描述
        doc_content = f"{video.get('title', '')}\n{video.get('description', '')}"
        
        documents.append(doc_content)
        metadatas.append({
            "title": video.get('title', ''),
            "url": video.get('url', ''),
            "fraud_type": video.get('fraud_type', '通用'),
            "description": video.get('description', ''),
            "content_type": "video"
        })
        ids.append(f"video_{i}")
    
    # 导入到 ChromaDB - videos 集合
    vector_db.videos_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(f"成功导入 {len(videos)} 条视频数据")


def create_sample_data():
    """创建示例数据文件（用于测试）"""
    # 示例案例数据
    sample_cases = [
        {
            "title": "冒充电商物流客服",
            "content": "昨天有个自称淘宝客服的人给我打电话，说我买的快递丢了，要给我双倍理赔，让我下载屏幕共享软件，最后转走两万。",
            "risk_level": "极高危",
            "target_group": "经常网购者,青年,宝妈",
            "fraud_type": "冒充客服诈骗"
        },
        {
            "title": "刷单返利骗局",
            "content": "群里有人发兼职广告，说刷一单返5元，我试了几单真的返了，后来让我刷大额单，说连刷三单才能返现，结果被骗5000元。",
            "risk_level": "高危",
            "target_group": "学生,宝妈,待业者",
            "fraud_type": "刷单返利诈骗"
        },
        {
            "title": "虚假投资理财",
            "content": "网上认识的投资导师带我炒股，先让我小额试水赚了，后来让我加大投入，平台突然无法提现，被骗20万。",
            "risk_level": "极高危",
            "target_group": "老人,中产",
            "fraud_type": "虚假投资理财诈骗"
        },
        {
            "title": "冒充公检法",
            "content": "接到自称公安局的电话，说我的银行卡涉嫌洗钱，要我配合调查，把资金转到安全账户，被骗10万元。",
            "risk_level": "极高危",
            "target_group": "老人",
            "fraud_type": "冒充公检法诈骗"
        },
        {
            "title": "杀猪盘网恋",
            "content": "在社交软件认识了一个高富帅，聊了很久确立恋爱关系，他说有内部投资渠道可以赚钱，让我一起投资，最后平台跑路。",
            "risk_level": "高危",
            "target_group": "单身女性,离异",
            "fraud_type": "杀猪盘网恋诈骗"
        }
    ]
    
    # 示例标语数据
    sample_slogans = [
        {"content": "陌生来电不轻信，转账汇款多核实", "fraud_type": "通用"},
        {"content": "刷单返利是陷阱，天上不会掉馅饼", "fraud_type": "刷单返利诈骗"},
        {"content": "公检法机关不会电话办案，更不会要求转账", "fraud_type": "冒充公检法诈骗"},
        {"content": "网恋对象带你投资，一定是诈骗", "fraud_type": "杀猪盘网恋诈骗"},
        {"content": "客服主动退款，多半是骗子", "fraud_type": "冒充客服诈骗"},
        {"content": "高额回报的投资理财，都是骗局", "fraud_type": "虚假投资理财诈骗"},
        {"content": "先交钱的贷款，都是诈骗", "fraud_type": "虚假贷款诈骗"},
        {"content": "游戏装备私下交易，小心钱号两空", "fraud_type": "游戏产品虚假交易"}
    ]
    
    # 示例视频数据
    sample_videos = [
        {
            "title": "揭秘冒充客服诈骗",
            "description": "详细讲解冒充电商客服诈骗的常见手法和防范技巧",
            "url": "/videos/impersonate_customer_service.mp4",
            "fraud_type": "冒充客服诈骗"
        },
        {
            "title": "刷单返利骗局揭秘",
            "description": "真实案例还原刷单诈骗全过程",
            "url": "/videos/brushing_scam.mp4",
            "fraud_type": "刷单返利诈骗"
        },
        {
            "title": "老年人防骗指南",
            "description": "针对老年人的常见诈骗类型及防范方法",
            "url": "/videos/elderly_anti_fraud.mp4",
            "fraud_type": "通用"
        }
    ]
    
    # 保存示例文件
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, 'sample_cases.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_cases, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(data_dir, 'sample_slogans.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_slogans, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(data_dir, 'sample_videos.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_videos, f, ensure_ascii=False, indent=2)
    
    logger.info(f"示例数据已创建在 {data_dir} 目录")
    return {
        "cases": os.path.join(data_dir, 'sample_cases.json'),
        "slogans": os.path.join(data_dir, 'sample_slogans.json'),
        "videos": os.path.join(data_dir, 'sample_videos.json')
    }


def main():
    parser = argparse.ArgumentParser(description='导入反诈推荐数据到向量数据库')
    parser.add_argument('--cases', help='案例数据 JSON 文件路径')
    parser.add_argument('--slogans', help='标语数据 JSON 文件路径')
    parser.add_argument('--videos', help='视频数据 JSON 文件路径')
    parser.add_argument('--create-samples', action='store_true', help='创建示例数据并导入')
    
    args = parser.parse_args()
    
    if args.create_samples:
        logger.info("创建示例数据...")
        sample_files = create_sample_data()
        import_cases(sample_files['cases'])
        import_slogans(sample_files['slogans'])
        import_videos(sample_files['videos'])
    else:
        if args.cases:
            import_cases(args.cases)
        if args.slogans:
            import_slogans(args.slogans)
        if args.videos:
            import_videos(args.videos)
    
    logger.info("数据导入完成！")


if __name__ == "__main__":
    main()
