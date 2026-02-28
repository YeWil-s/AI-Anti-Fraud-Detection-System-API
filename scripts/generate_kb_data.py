"""
反诈知识库初始化脚本 (大赛正式版)
读取清洗后的本地多模态案例数据集 (JSON)，灌入 ChromaDB 向量数据库
"""
import sys
import os
import json

# 将项目根目录加入环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.services.vector_db_service import vector_db
from app.core.logger import get_logger

logger = get_logger(__name__)

DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "processed_cases.json")

def init_db():
    print("====== 开始初始化反诈知识库 ======")
    
    # 1. 检查数据文件是否存在
    if not os.path.exists(DATA_FILE_PATH):
        logger.error(f"找不到数据文件: {DATA_FILE_PATH}")
        print("请先按照大赛要求，将收集清洗后的 20+ 个案例存入 data/processed_cases.json 中！")
        return

    # 2. 读取 JSON 数据
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    if len(cases) < 20:
        logger.warning(f"当前案例数量为 {len(cases)}，未达到大赛要求的不少于 20 个！")
        
    documents = []
    metadatas = []
    ids = []
    
    # 3. 解析并构建向量库入库格式
    for case in cases:
        ids.append(case.get("id"))
        documents.append(case.get("content"))
        
        # 提取 metadata，移除大段的 content 文本以节省元数据空间
        meta = {
            "modality": case.get("modality", "text"),
            "fraud_type": case.get("fraud_type", "未知"),
            "risk_level": case.get("risk_level", "未知"),
            "source": case.get("source", "未知")
        }
        # 如果有音视频文件路径，也存入 metadata
        if "file_path" in case:
            meta["file_path"] = case["file_path"]
            
        metadatas.append(meta)

    # 4. 执行批量插入
    try:
        vector_db.add_cases(documents, metadatas, ids)
        print(f"成功将 {len(ids)} 个真实反诈案例灌入向量数据库！")
    except Exception as e:
        logger.error(f"向量数据库写入失败: {e}", exc_info=True)
        print(f"写入失败: {e}")

    # 5. 简单检索测试
    print("\n====== 执行 RAG 检索测试 ======")
    test_query = "我是警察，你涉嫌洗钱，请把钱转到安全账户"
    print(f"模拟用户被骗输入: '{test_query}'\n")
    
    results = vector_db.search_similar_cases(test_query, n_results=1)
    if results['documents'] and results['documents'][0]:
        print("【检索命中】:")
        print(f"- 匹配案例: {results['documents'][0][0]}")
        print(f"- 诈骗类型: {results['metadatas'][0][0]['fraud_type']}")
        print(f"- 模态来源: {results['metadatas'][0][0]['modality']}")
        print(f"- 官方数据源: {results['metadatas'][0][0]['source']}")

if __name__ == "__main__":
    init_db()