"""
反诈知识库初始化与数据增强脚本 (大赛高分版)
1. 读取组员手工清洗的种子数据集 (processed_cases.json)
2. 调用 LLM 自动扩写 200+ 个不同维度的变体话术
3. 统一灌入 ChromaDB 向量数据库
"""
import sys
import os
import json
import asyncio
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field

# 将项目根目录加入环境变量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.services.vector_db_service import vector_db
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)

DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "processed_cases.json")

# 强制 LLM 输出的结构
class CaseVariations(BaseModel):
    variations: List[str] = Field(description="生成的诈骗话术变体列表，包含具体的对话内容或场景描述")

class KnowledgeBaseGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0.7, # 稍微高一点的温度，保证变体的多样性
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.output_parser = JsonOutputParser(pydantic_object=CaseVariations)
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的反诈数据生成专家。
你的任务是根据提供的【种子案例】，生成 {num_variations} 个该诈骗类型的【变体话术】。
要求：
1. 变体必须覆盖不同的受害人群（如老人、学生、宝妈）、不同的沟通平台（如微信、电话、短视频直播间）。
2. 话术要尽量逼真，口语化，包含诱导转账、下载APP、索要验证码等核心诈骗要素。
3. 如果种子案例是音视频，请生成对应的【语音转写】或【视频画面描述】。
4. 严格按照 JSON 格式输出列表。

{format_instructions}
"""),
            ("human", "种子案例类型：{fraud_type}\n种子案例内容：{content}")
        ])

    async def generate_variations(self, fraud_type: str, content: str, num_variations: int = 10) -> List[str]:
        """调用大模型生成变体"""
        try:
            chain = self.prompt_template | self.llm | self.output_parser
            response = await chain.ainvoke({
                "fraud_type": fraud_type,
                "content": content,
                "num_variations": num_variations,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            return response.get("variations", [])
        except Exception as e:
            logger.error(f"生成 {fraud_type} 变体失败: {e}")
            return []

async def init_and_augment_db():
    print("====== 开始数据增强与知识库初始化 ======")
    
    if not os.path.exists(DATA_FILE_PATH):
        logger.error(f"找不到种子数据文件: {DATA_FILE_PATH}")
        return

    with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    if len(cases) < 20:
        logger.warning(f"当前种子案例仅有 {len(cases)} 个！")

    generator = KnowledgeBaseGenerator()
    
    all_documents = []
    all_metadatas = []
    all_ids = []
    
    case_counter = 0

    print("🚀 正在通过大模型扩写案例，请耐心等待（可能需要几分钟）...")
    
    # 1. 遍历种子案例并扩写
    for seed_case in cases:
        fraud_type = seed_case.get("fraud_type", "未知")
        modality = seed_case.get("modality", "text")
        content = seed_case.get("content", "")
        
        # 先把种子案例加进去
        all_documents.append(content)
        all_metadatas.append({
            "modality": modality,
            "fraud_type": fraud_type,
            "risk_level": seed_case.get("risk_level", "未知"),
            "source": seed_case.get("source", "原始种子")
        })
        all_ids.append(f"case_seed_{case_counter}")
        case_counter += 1

        # 针对每个黑样本（诈骗），生成 10 个变体
        # 如果是白样本（安全），可以少生成几个，或者不生成
        if seed_case.get("risk_level") in ["高危", "极高危"]:
            print(f"正在扩写: [{fraud_type}] ...")
            variations = await generator.generate_variations(fraud_type, content, num_variations=10)
            
            for var_content in variations:
                all_documents.append(var_content)
                all_metadatas.append({
                    "modality": modality, # 继承原模态描述方式
                    "fraud_type": fraud_type,
                    "risk_level": seed_case.get("risk_level", "未知"),
                    "source": "LLM_Augmented"
                })
                all_ids.append(f"case_var_{case_counter}")
                case_counter += 1

    # 2. 灌入 ChromaDB
    print(f"\n📦 数据扩写完成，共准备入库 {len(all_ids)} 条数据（含种子与变体）。")
    try:
        # 这里假设你的 vector_db.add_cases 方法支持直接插入
        vector_db.add_cases(all_documents, all_metadatas, all_ids)
        print("✅ 成功灌入向量数据库！")
    except Exception as e:
        logger.error(f"写入向量库失败: {e}", exc_info=True)
        print("❌ 写入失败，请检查日志。")

    # 3. 检索测试
    print("\n====== 执行 RAG 检索测试 ======")
    test_query = "领导让我立刻给这个对公账户打钱，说有急用"
    print(f"模拟用户被骗输入: '{test_query}'\n")
    
    results = vector_db.search_similar_cases(test_query, n_results=1)
    if results and results.get('documents') and results['documents'][0]:
        print("【检索命中】:")
        print(f"- 匹配案例: {results['documents'][0][0]}")
        print(f"- 诈骗类型: {results['metadatas'][0][0]['fraud_type']}")
        print(f"- 官方数据源: {results['metadatas'][0][0]['source']}")

if __name__ == "__main__":
    asyncio.run(init_and_augment_db())