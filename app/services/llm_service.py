"""
LLM 大模型服务 (Agent 的大脑)
负责接收文本、查询 RAG 知识库、拼接 Prompt 并输出结构化 JSON 结果
"""
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.services.vector_db_service import vector_db

logger = get_logger(__name__)

# 定义大模型强制输出的 JSON 结构
class RiskAssessmentOutput(BaseModel):
    is_fraud: bool = Field(description="是否判定为诈骗风险")
    risk_level: str = Field(description="风险等级：safe, medium, high, critical")
    confidence: float = Field(description="置信度评分，0.0 到 1.0 之间")
    analysis: str = Field(description="对用户输入内容的分析过程和意图识别")
    advice: str = Field(description="针对特定受众人群（如老人、学生）给出的个性化防骗建议")

class LLMService:
    def __init__(self):
        # 推荐使用智谱 GLM-4 或 DeepSeek，配置填在 .env 中
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME, # 例如 "glm-4" 或 "deepseek-chat"
            temperature=0.1, # 降低温度，保证决策稳定性
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.output_parser = JsonOutputParser(pydantic_object=RiskAssessmentOutput)

        # 核心：多模态反诈智能体系统提示词
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个国家级的反诈智能体助手，具备强大的意图识别与推理能力。
你的任务是保护用户免受电信网络诈骗。

【当前用户的角色画像】：{role_type}
注意：请严格根据用户的角色画像动态调整你的风险容忍度和审查重点。以下是不同人群的易受骗场景，当输入内容触发对应场景时，请直接提高风险等级：
- 针对“老人”：要对推销特效药/保健品、高息养老投资理财、冒充公检法恐吓扣款、以及冒充亲友遭遇车祸/疾病等紧急情况要求转账的信息极度敏感。
- 针对“学生”：要对网络兼职刷单赚佣金、声称注销校园贷/修复不良征信、非官方渠道低价售卖演唱会门票、以及游戏账号/装备私下交易类信息高度警惕。
- 针对“孩子（儿童）”：要对免费领取游戏皮肤/道具、承诺解除防沉迷系统、诱导索要父母手机验证码/扫码付款、以及冒充网警恐吓要抓捕父母的信息实施“零容忍”级别的拦截。
- 针对“青壮年”：要对所谓内部渠道的高回报投资（杀猪盘）、无需征信的虚假网络贷款平台、冒充电商或快递客服主动理赔退款、以及冒充公司领导/老板要求紧急过桥垫资类信息保持严密防范。
【知识库参考案例（RAG检索结果）】：
{context}

请分析用户的输入，判断是否存在诈骗风险，并严格按照 JSON 格式输出结果。
{format_instructions}"""),
            ("human", "【用户输入内容】：{user_input}")
        ])

    async def analyze_text_risk(self, user_input: str, role_type: str = "青壮年") -> dict:
        """
        全链路文本风控分析
        """
        try:
            # 1. 检索 ChromaDB 获取相似案例 (Day 2 的成果)
            context = vector_db.get_context_for_llm(user_input, n_results=2)
            
            # 2. 构建处理链条 (Chain)
            chain = self.prompt_template | self.llm | self.output_parser
            
            # 3. 异步调用大模型
            response = await chain.ainvoke({
                "role_type": role_type,
                "context": context,
                "user_input": user_input,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            logger.info(f"LLM Assessment Success: {response['risk_level']} | Fraud: {response['is_fraud']}")
            return response
            
        except Exception as e:
            logger.error(f"LLM Service Error: {e}")
            # 降级处理：模型报错时默认返回安全
            return {
                "is_fraud": False, "risk_level": "safe", "confidence": 0.0,
                "analysis": "大模型调用异常，降级通过", "advice": "系统繁忙"
            }

llm_service = LLMService()