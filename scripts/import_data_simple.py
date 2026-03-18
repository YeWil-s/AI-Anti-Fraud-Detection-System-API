"""
简化版案例库和法律库导入脚本
直接使用 ChromaDB 客户端
"""
import os
import sys
import json
import hashlib
from datetime import datetime

# 禁用遥测
os.environ['ANONYMIZED_TELEMETRY'] = 'false'
os.environ['CHROMA_TELEMETRY'] = 'false'

import chromadb
from chromadb.utils import embedding_functions

# ChromaDB 数据目录
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")


def get_chroma_client():
    """获取 ChromaDB 客户端"""
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )


def get_embedding_function():
    """获取中文embedding函数"""
    # 使用轻量级的中文embedding模型
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )


def normalize_fraud_type(fraud_type: str) -> str:
    """标准化诈骗类型"""
    if not fraud_type:
        return "other"
    
    mapping = {
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
    }
    
    fraud_type = fraud_type.strip()
    return mapping.get(fraud_type, fraud_type.lower().replace(" ", "_"))


def import_cases(file_path: str, collection, source_type: str = "case_library"):
    """导入案例或法律条文"""
    print(f"\n导入文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"  错误: 数据格式不正确")
        return 0
    
    imported = 0
    documents = []
    metadatas = []
    ids = []
    
    for i, item in enumerate(data):
        try:
            title = item.get("title", "")
            content = item.get("content", "")
            
            if not content:
                continue
            
            # 构建文档
            doc_text = f"{title}\n\n{content}" if title else content
            
            # 提取元数据
            fraud_type = normalize_fraud_type(item.get("fraud_type", ""))
            
            metadata = {
                "title": title,
                "fraud_type": fraud_type,
                "source": source_type,
                "imported_at": datetime.now().isoformat()
            }
            
            # 添加目标人群（如果是案例）
            if "target_group" in item:
                metadata["target_groups"] = item["target_group"]
            
            # 添加风险等级（如果是案例）
            if "risk_level" in item:
                metadata["risk_level"] = item["risk_level"]
            
            # 生成唯一ID
            doc_id = hashlib.md5(f"{source_type}_{i}_{title[:50]}".encode()).hexdigest()
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(doc_id)
            
            imported += 1
            
            # 每50条批量导入一次
            if len(documents) >= 50:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"  已导入 {imported} 条...")
                documents = []
                metadatas = []
                ids = []
                
        except Exception as e:
            print(f"  导入第 {i} 条失败: {e}")
            continue
    
    # 导入剩余数据
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    print(f"  完成，共导入 {imported} 条")
    return imported


def main():
    """主函数"""
    print("=" * 60)
    print("案例库和法律库导入工具")
    print("=" * 60)
    
    # 获取 ChromaDB 客户端
    client = get_chroma_client()
    embedding_func = get_embedding_function()
    
    # 获取或创建集合
    try:
        collection = client.get_collection("anti_fraud_cases")
        print("\n使用已存在的集合: anti_fraud_cases")
    except:
        collection = client.create_collection(
            name="anti_fraud_cases",
            embedding_function=embedding_func,
            metadata={"description": "反诈案例库和法律库"}
        )
        print("\n创建新集合: anti_fraud_cases")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cases_dir = os.path.join(base_dir, "data", "competition_cases")
    
    total = 0
    
    # 导入案例库
    case_files = [
        ("300数据集标注.json", "case_library"),
        ("300数据集标注 (2).json", "case_library")
    ]
    
    for filename, source in case_files:
        filepath = os.path.join(cases_dir, filename)
        if os.path.exists(filepath):
            count = import_cases(filepath, collection, source)
            total += count
        else:
            print(f"\n文件不存在: {filepath}")
    
    # 导入法律库
    law_files = [
        ("法律234.json", "law_library"),
        ("法律567.json", "law_library")
    ]
    
    for filename, source in law_files:
        filepath = os.path.join(cases_dir, filename)
        if os.path.exists(filepath):
            count = import_cases(filepath, collection, source)
            total += count
        else:
            print(f"\n文件不存在: {filepath}")
    
    print("\n" + "=" * 60)
    print(f"导入完成！总计: {total} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
