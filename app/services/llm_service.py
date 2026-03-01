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
            ("system", """你是一个国家级的多模态反诈智能体助手，具备强大的意图识别与复杂逻辑推理能力。
你的任务是综合分析【用户对话文本】和【底层AI鉴伪模型的视觉/听觉特征得分】，保护用户免受欺诈。

【当前用户的角色画像】：{role_type}
注意：请严格根据用户的角色画像动态调整你的风险容忍度、审查重点和提醒的方式。以下是不同人群的易受骗场景：
- 老人：对推销特效药、高息养老理财、冒充公检法恐吓、冒充亲友要求转账极度敏感。
- 学生：对兼职刷单、注销校园贷、非官方低价票、游戏账号交易高度警惕。
- 孩子：对免费领游戏皮肤、解除防沉迷、诱导索要父母验证码零容忍。
- 青壮年：对内部高回报投资(杀猪盘)、虚假网贷、冒充老板过桥垫资保持防范。

【知识库参考案例（RAG检索结果）】：
{context}

【底层AI模型多模态特征分析结果】（极其重要！）：
- 语音伪造(Voice Clone)置信度：{audio_conf} （0~1.0，>0.7通常代表极有可能是AI合成变声）
- 视频伪造(Deepfake)置信度：{video_conf} （0~1.0，>0.75通常代表画面经过AI换脸或唇形篡改）

【多模态融合决策规则】：
1. 交叉验证：如果文本内容是“我是你领导/亲人，快打钱”，且【语音/视频伪造置信度】处于高位，这必定是极其危险的 AI 拟声/换脸诈骗，必须判定为 is_fraud=True，risk_level="critical"。
2. 纠正误报：如果【语音置信度】较高（比如0.72），但对话文本完全是正常业务（例如快递员送件“喂你的快递到了”），请结合常识判定为安全，纠正底层音频模型的误报。

请综合推理，并严格按照 JSON 格式输出最终裁定结果。
{format_instructions}"""),
            ("human", "【用户输入内容(含语音转写)】：{user_input}")
        ])

    async def analyze_text_risk(self, user_input: str, role_type: str = "青壮年") -> dict:
        """
        全链路文本风控分析
        """
        try:
            # 1. 检索 ChromaDB 获取相似案例
            context = vector_db.get_context_for_llm(user_input, n_results=2)
            
            # 2. 构建处理链条 
            chain = self.prompt_template | self.llm | self.output_parser
            
            # 3. 异步调用大模型
            response = await chain.ainvoke({
                "role_type": role_type,
                "context": context,
                "user_input": user_input,
                "audio_conf": f"{audio_conf:.4f}",
                "video_conf": f"{video_conf:.4f}",
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            logger.info(f"融合决策完成: 判定诈骗={response['is_fraud']} | 等级={response['risk_level']} | A_conf={audio_conf:.2f} | V_conf={video_conf:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"LLM 融合分析异常: {e}")
            # 降级处理：模型报错时默认返回安全
            return {
                "is_fraud": False, "risk_level": "safe", "confidence": 0.0,
                "analysis": "大模型调用异常，降级通过", "advice": "系统繁忙"
            }

llm_service = LLMService()