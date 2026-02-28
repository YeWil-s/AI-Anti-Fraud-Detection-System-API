"""
向量数据库服务 (RAG 检索基础)
用于存储和检索反诈骗典型案例
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
        
        # 获取或创建集合 (Collection，类似于关系型数据库中的表)
        self.collection = self.client.get_or_create_collection(
            name="anti_fraud_cases",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度进行检索
        )
        logger.info("ChromaDB Vector DB initialized successfully.")

    def add_cases(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """
        向知识库中添加新的反诈案例
        :param documents: 案例正文列表
        :param metadatas: 元数据列表（如案件类型、危险等级）
        :param ids: 唯一标识符列表
        """
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Successfully upserted {len(ids)} cases into Vector DB.")

    def search_similar_cases(self, query: str, n_results: int = 3) -> dict:
        """
        根据用户输入，检索最相似的历史案例
        :param query: 用户的聊天内容或风险文本
        :param n_results: 返回的最相似案例数量
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def get_context_for_llm(self, query: str, n_results: int = 3) -> str:
        """
        [补充增强] 将检索到的相似案例组装成一段易于大模型阅读的纯文本上下文
        """
        results = self.search_similar_cases(query, n_results=n_results)
        
        if not results['documents'] or not results['documents'][0]:
            return "未在知识库中检索到相似案例。"
            
        context_parts = []
        # 遍历检索到的结果
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # 距离越小越相似，设定一个合理的阈值过滤掉不相关的结果(余弦距离 < 0.6 表示相关性较高)
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