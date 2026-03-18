"""
将法律数据从案例集合迁移到单独的法律集合
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
    print("迁移法律数据到单独集合")
    print("=" * 60)
    
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    
    # 获取集合
    case_collection = client.get_collection("anti_fraud_cases")
    law_collection = client.get_collection("anti_fraud_laws")
    
    print("\n1. 从案例集合中查询法律数据...")
    
    # 查询所有数据
    all_data = case_collection.get()
    
    if not all_data or not all_data['ids']:
        print("  案例集合为空")
        return
    
    total = len(all_data['ids'])
    print(f"  案例集合共有 {total} 条数据")
    
    # 筛选出法律数据
    law_ids = []
    law_documents = []
    law_metadatas = []
    
    for i, doc_id in enumerate(all_data['ids']):
        metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
        source = metadata.get('source', '')
        
        if source == 'law_library':
            law_ids.append(doc_id)
            law_documents.append(all_data['documents'][i])
            law_metadatas.append(metadata)
    
    print(f"\n2. 找到 {len(law_ids)} 条法律数据")
    
    if law_ids:
        print("\n3. 导入到法律集合...")
        
        # 分批导入
        batch_size = 50
        for i in range(0, len(law_ids), batch_size):
            batch_ids = law_ids[i:i+batch_size]
            batch_docs = law_documents[i:i+batch_size]
            batch_meta = law_metadatas[i:i+batch_size]
            
            law_collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )
            print(f"  已导入 {min(i+batch_size, len(law_ids))}/{len(law_ids)} 条")
        
        print(f"\n✅ 成功导入 {len(law_ids)} 条法律数据到 anti_fraud_laws")
        
        # 可选：从案例集合中删除法律数据
        # print("\n4. 从案例集合中删除法律数据...")
        # case_collection.delete(ids=law_ids)
        # print(f"✅ 已删除案例集合中的 {len(law_ids)} 条法律数据")
        
    print("\n" + "=" * 60)
    print("迁移完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
