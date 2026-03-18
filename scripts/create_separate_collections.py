"""
创建单独的法律库集合，将案例和法律分开存储
"""
import os
import sys

os.environ['ANONYMIZED_TELEMETRY'] = 'false'
os.environ['CHROMA_TELEMETRY'] = 'false'

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_data")


def main():
    print("=" * 60)
    print("创建单独的法律库集合")
    print("=" * 60)
    
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 检查是否已存在
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    
    print(f"\n现有集合: {collection_names}")
    
    # 创建法律库集合
    if "anti_fraud_laws" not in collection_names:
        law_collection = client.create_collection(
            name="anti_fraud_laws",
            embedding_function=embedding_func,
            metadata={"description": "反诈法律法规库"}
        )
        print("✅ 创建集合: anti_fraud_laws")
    else:
        print("✅ 集合已存在: anti_fraud_laws")
    
    # 确保案例库集合存在
    if "anti_fraud_cases" not in collection_names:
        case_collection = client.create_collection(
            name="anti_fraud_cases",
            embedding_function=embedding_func,
            metadata={"description": "反诈案例库"}
        )
        print("✅ 创建集合: anti_fraud_cases")
    else:
        print("✅ 集合已存在: anti_fraud_cases")
    
    print("\n" + "=" * 60)
    print("集合创建完成！")
    print("  - anti_fraud_cases: 存储案例")
    print("  - anti_fraud_laws: 存储法律条文")
    print("=" * 60)


if __name__ == "__main__":
    main()
