"""
向量数据库服务 (RAG 检索基础)
用于存储和检索反诈骗典型案例与法律法规
"""
import os
from chromadb.config import Settings
import chromadb
from chromadb.utils import embedding_functions
from app.core.logger import get_logger

logger = get_logger(__name__)

# 获取项目根目录，在根目录下创建一个 chroma_data 文件夹用于持久化存储向量数据
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DATA_PATH = os.path.join(BASE_DIR, "chroma_data")

class VectorDBService:
    def __init__(self):
        # 初始化持久化客户端（重启后数据不丢失）
        self.client = chromadb.PersistentClient(
            path=CHROMA_DATA_PATH,
            settings=Settings(anonymized_telemetry=False))
        
        # 使用轻量级开源多语言 Embedding 模型
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 1. 典型案例库 (原有)
        self.cases_collection = self.client.get_or_create_collection(
            name="anti_fraud_cases",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度进行检索
        )
        
        # 2. 法律法规库 (新增)
        self.laws_collection = self.client.get_or_create_collection(
            name="anti_fraud_laws",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} 
        )
        
        logger.info("ChromaDB Vector DB initialized successfully with cases and laws collections.")

    # ================= 原有兼容方法 =================
    def add_cases(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """向知识库中添加新的反诈案例 (兼容老代码)"""
        self.cases_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Successfully upserted {len(ids)} cases into Vector DB.")

    def search_similar_cases(self, query: str, n_results: int = 3) -> dict:
        """根据用户输入，检索最相似的历史案例 (兼容老代码返回原生格式)"""
        return self.cases_collection.query(query_texts=[query], n_results=n_results)
    
    # ================= 新增通用方法 =================
    def add_data(self, collection_name: str, documents: list[str], metadatas: list[dict], ids: list[str]):
        """通用数据入库方法"""
        collection = self.cases_collection if collection_name == "anti_fraud_cases" else self.laws_collection
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Successfully upserted {len(ids)} items into {collection_name}.")

    def search_similar(self, collection_name: str, text: str, top_k: int = 2) -> list:
        """
        通用检索方法 (供 EducationService 使用)
        返回格式化好的字典列表
        """
        collection = self.cases_collection if collection_name == "anti_fraud_cases" else self.laws_collection
        results = collection.query(
            query_texts=[text],
            n_results=top_k
        )
        
        formatted_results = []
        if results and results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if 'distances' in results and results['distances'] else None
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": distance
                })
        return formatted_results

    def get_context_for_llm(self, query: str, n_results: int = 3) -> str:
        """将检索到的相似案例组装成一段易于大模型阅读的纯文本上下文"""
        results = self.search_similar_cases(query, n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return "未在知识库中检索到相似案例。"
            
        context_parts = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            if distance > 0.6:
                continue
                
            fraud_type = meta.get('fraud_type', '未知')
            risk_level = meta.get('risk_level', '未知')
            
            case_text = f"案例{i+1}: [类型: {fraud_type}] [风险等级: {risk_level}]\n内容: {doc}\n"
            context_parts.append(case_text)
            
        if not context_parts:
            return "检索到的案例相关性较低，无参考价值。"
            
        return "\n".join(context_parts)

# 实例化单例，供其他模块引入使用
vector_db = VectorDBService()