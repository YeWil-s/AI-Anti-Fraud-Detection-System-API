"""
反诈知识库初始化脚本
读取清洗后的本地案例数据集 (JSON)，灌入 ChromaDB 向量数据库
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
        print("请先按照要求，将清洗后的案例存入 data/processed_cases.json 中！")
        return

    # 2. 读取 JSON 数据
    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    if len(cases) < 20:
        logger.warning(f"当前案例数量为 {len(cases)}，未达到大赛要求的不少于 20 个！")
        
    documents = []
    metadatas = []
    ids = []
    
    # 3. 解析并构建向量库入库格式 (修改点：自动生成ID，适配新标签)
    for i, case in enumerate(cases):
        # 自动生成唯一 ID (如 case_0, case_1)
        ids.append(f"case_{i}")
        documents.append(case.get("content"))
        
        # 提取 metadata，只保留核心标签和新增的受众群体标签
        meta = {
            "fraud_type": case.get("fraud_type", "未知"),
            "risk_level": case.get("risk_level", "未知"),
            "target_group": case.get("target_group", "通用人群")
        }
        # 如果你未来有音视频文件路径，依然可以保留这段逻辑
        if "file_path" in case:
            meta["file_path"] = case["file_path"]
            
        metadatas.append(meta)

    # 4. 执行批量插入
    try:
        # 使用我们之前重构好的 add_data 方法（或者你用旧的 add_cases 也可以）
        vector_db.add_data(
            collection_name="anti_fraud_cases", 
            documents=documents, 
            metadatas=metadatas, 
            ids=ids
        )
        print(f"成功将 {len(ids)} 个真实反诈案例灌入向量数据库！")
    except Exception as e:
        logger.error(f"向量数据库写入失败: {e}", exc_info=True)
        print(f"写入失败: {e}")

    # 5. 简单检索测试 (修改点：去掉了已删除字段的打印，增加新字段打印)
    print("\n====== 执行 RAG 检索测试 ======")
    test_query = "我是警察，你涉嫌洗钱，请把钱转到安全账户"
    print(f"模拟用户被骗输入: '{test_query}'\n")
    
    # 使用我们刚写好的 search_similar 字典列表返回格式
    results = vector_db.search_similar("anti_fraud_cases", test_query, top_k=1)
    
    if results:
        best_match = results[0]
        print("【检索命中】:")
        print(f"- 匹配案例: {best_match['content']}")
        print(f"- 诈骗类型: {best_match['metadata'].get('fraud_type', '未知')}")
        print(f"- 目标人群: {best_match['metadata'].get('target_group', '未知')}")
        print(f"- 风险等级: {best_match['metadata'].get('risk_level', '未知')}")
    else:
        print("未检索到相关案例。")

if __name__ == "__main__":
    init_db()