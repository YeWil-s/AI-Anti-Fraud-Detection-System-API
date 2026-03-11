import os
import json
import uuid
import sys

# 确保能导入 app 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_db_service import vector_db

def load_json_to_rag(json_file_path: str):
    """读取 JSON 文件并存入向量数据库"""
    if not os.path.exists(json_file_path):
        print(f"❌ 找不到文件: {json_file_path}")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    source_name = data.get("website_name", "未知来源")
    cases = data.get("source_data", [])
    
    documents = []
    metadatas = []
    ids = []
    
    for case in cases:
        title = case.get("title", "")
        content = case.get("content", "")
        
        # 1. 组装送入大模型的文本内容
        # 将标题和内容合并，提供更丰富的语义
        full_text = f"【标题】{title}\n【套路详情】{content}"
        
        # 2. 提取元数据 (Metadata)
        # 可以用简单的规则提取 fraud_type，或者直接用标题前几个字
        fraud_type = title[:10] + "..." if len(title) > 10 else title
        
        meta = {
            "source": source_name,
            "fraud_type": fraud_type,
            "risk_level": "高危" # 默认高危，也可以根据内容动态打标
        }
        
        documents.append(full_text)
        metadatas.append(meta)
        ids.append(str(uuid.uuid4())) # 生成唯一 ID
        
    # 批量写入 ChromaDB
    if documents:
        print(f"📦 正在将 {len(documents)} 条案例写入 RAG 知识库 (来源: {source_name})...")
        vector_db.add_cases(documents=documents, metadatas=metadatas, ids=ids)
        print("✅ 写入完成！\n")

if __name__ == "__main__":
    # 假设你把这四个 json 文件放在了 data/competition_cases/ 目录下
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "competition_cases")
    
    # 如果目录不存在，请自行创建并把 json 扔进去
    os.makedirs(data_dir, exist_ok=True)
    
    files_to_import = [
        "搜狗反诈.json",
        "搜狗-诈骗套路.json",
        "搜狗诈骗预警.json",
        "百度反诈.json"
    ]
    
    for filename in files_to_import:
        file_path = os.path.join(data_dir, filename)
        load_json_to_rag(file_path)
        
    print("大赛提供的所有反诈物料已成功注入 RAG 系统！")