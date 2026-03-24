"""
向量数据库服务 (RAG 检索基础)
用于存储和检索反诈骗典型案例与法律法规
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from chromadb.config import Settings
import chromadb
from chromadb.utils import embedding_functions
from app.core.logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DATA_PATH = os.path.join(BASE_DIR, "chroma_data")

class VectorDBService:
    def __init__(self):
        # 初始化持久化客户端
        self.client = chromadb.PersistentClient(
            path=CHROMA_DATA_PATH,
            settings=Settings(anonymized_telemetry=False))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 1. 典型案例库
        self.cases_collection = self.client.get_or_create_collection(
            name="anti_fraud_cases",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度进行检索
        )
        
        # 2. 法律法规库
        self.laws_collection = self.client.get_or_create_collection(
            name="anti_fraud_laws",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} 
        )
        
        # 3. 宣传标语库 
        self.slogans_collection = self.client.get_or_create_collection(
            name="anti_fraud_slogans",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 4. 宣传视频库 
        self.videos_collection = self.client.get_or_create_collection(
            name="anti_fraud_videos",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("ChromaDB Vector DB initialized successfully with cases, laws, slogans and videos collections.")

    def add_cases(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """向知识库中添加新的反诈案例"""
        self.cases_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Successfully upserted {len(ids)} cases into Vector DB.")

    def search_similar_cases(self, query: str, n_results: int = 3) -> dict:
        """根据用户输入，检索最相似的历史案例"""
        return self.cases_collection.query(query_texts=[query], n_results=n_results)

    def add_data(self, collection_name: str, documents: list[str], metadatas: list[dict], ids: list[str]):
        """通用数据入库方法"""
        collection = self.cases_collection if collection_name == "anti_fraud_cases" else self.laws_collection
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(f"Successfully upserted {len(ids)} items into {collection_name}.")

    def _get_collection(self, collection_name: str):
        """根据名称获取对应的集合"""
        collections = {
            "anti_fraud_cases": self.cases_collection,
            "anti_fraud_laws": self.laws_collection,
            "anti_fraud_slogans": self.slogans_collection,
            "anti_fraud_videos": self.videos_collection,
        }
        return collections.get(collection_name, self.cases_collection)
    
    def search_similar(self, collection_name: str, text: str, top_k: int = 2, 
                       filter_dict: dict = None) -> list:
        """
        通用检索方法
        返回格式化好的字典列表
        
        Args:
            collection_name: 集合名称
            text: 查询文本
            top_k: 返回结果数量
            filter_dict: 过滤条件，如 {"fraud_type": "刷单返利诈骗"}
        """
        collection = self._get_collection(collection_name)
        
        query_params = {
            "query_texts": [text],
            "n_results": top_k
        }
        
        # 添加过滤条件
        if filter_dict:
            query_params["where"] = filter_dict
        
        results = collection.query(**query_params)
        
        formatted_results = []
        if results and results.get('documents') and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i] if 'distances' in results and results['distances'] else None
                similarity = 1 - distance if distance is not None else None
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": distance,
                    "similarity": round(similarity, 4) if similarity else None
                })
        return formatted_results
    
    def search_by_fraud_type(self, collection_name: str, fraud_type: str, top_k: int = 5) -> list:
        """
        根据诈骗类型搜索相关内容
        
        Args:
            collection_name: 集合名称
            fraud_type: 诈骗类型
            top_k: 返回数量
        """
        collection = self._get_collection(collection_name)
        
        results = collection.get(
            where={"fraud_type": fraud_type},
            limit=top_k
        )
        
        formatted_results = []
        if results and results.get('documents'):
            for i in range(len(results['documents'])):
                formatted_results.append({
                    "id": results['ids'][i],
                    "content": results['documents'][i],
                    "metadata": results['metadatas'][i]
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