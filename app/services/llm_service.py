"""
LLM 大模型服务 (Agent 的大脑)
负责接收文本、查询 RAG 知识库、拼接 Prompt 并输出结构化 JSON 结果
"""
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field 
from typing import List, Tuple, Optional
from app.core.config import settings
from app.core.logger import get_logger
from app.services.vector_db_service import vector_db
from app.models.user import User
from app.models.call_record import CallRecord
from app.models.ai_detection_log import AIDetectionLog
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger(__name__)

# 定义大模型强制输出的 JSON 结构
class RiskAssessmentOutput(BaseModel):
    is_fraud: bool = Field(description="是否判定为诈骗风险")
    risk_level: str = Field(description="风险等级：'safe', 'suspicious', 'fake'")
    confidence: float = Field(description="置信度评分，0.0 到 1.0 之间")
    analysis: str = Field(description="对用户输入内容的分析过程和意图识别")
    advice: str = Field(description="针对特定受众人群（如老人、学生）给出的个性化防骗建议")

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME, 
            temperature=0.1, 
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.output_parser = JsonOutputParser(pydantic_object=RiskAssessmentOutput)

        # 核心：多模态反诈智能体系统提示词 (包含 chat_history 记忆池槽位)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个国家级的多模态反诈智能体助手，具备强大的意图识别与复杂逻辑推理能力。
你的任务是综合分析【近期对话上下文】、【当前最新输入】和【底层AI特征得分】，保护用户免受欺诈。

【当前用户的角色画像】：{role_type}
注意：请严格根据用户的角色画像动态调整你的风险容忍度、审查重点。以下是不同人群的易受骗场景：
- 老人：对推销特效药、高息理财、冒充公检法恐吓、冒充亲友要求转账极度敏感。
- 学生：对兼职刷单、注销校园贷、非官方低价票、游戏账号交易高度警惕。
- 孩子：对免费领游戏皮肤、解除防沉迷、诱导索要父母验证码零容忍。
- 青壮年：对内部高回报投资(杀猪盘)、虚假网贷、冒充老板过桥垫资保持防范。

【知识库参考案例（RAG检索结果）】：
{context}

【近期对话上下文】（请结合这些历史记录来理解当前输入的真实意图）：
{chat_history}

【底层AI模型多模态特征分析结果】（极其重要！）：
- 语音伪造(Voice Clone)置信度：{audio_conf} （0~1.0，>0.7通常代表极有可能是AI合成变声）
- 视频伪造(Deepfake)置信度：{video_conf} （0~1.0，>0.75通常代表画面经过AI换脸或唇形篡改）

【多模态融合决策规则】：
1. 上下文连贯性：骗子通常会将“我是领导”、“出事了”、“快打钱”分段发送。请务必将【当前最新输入】与【近期对话上下文】结合看，一旦发现连环套话术，立即提高风险等级。
2. 交叉验证：如果文本内容涉及要钱/转账，且【语音/视频伪造置信度】处于高位，必须判定为 is_fraud=True，risk_level="fake"。
3. 纠正误报：如果语音置信度较高，但结合上下文完全是正常业务（如快递员送件），请结合常识判定为 safe，纠正底层误报。

请综合推理，并严格按照 JSON 格式输出最终裁定结果。
{format_instructions}

【极其重要】：你的输出必须且只能是一个合法的 JSON 对象。绝对不能包含任何 Markdown 语法（如 ```json ），不要包含任何解释性文本、不要有任何前言或后语。必须以大括号 {{ 开始，以大括号 }} 结束。"""),
            ("human", "【当前最新输入(含语音转写)】：{user_input}")
        ])

    async def analyze_text_risk(self, user_input: str, chat_history: str, role_type: str = "青壮年") -> dict:
        """纯文本风控分析（带记忆池）"""
        try:
            context = await asyncio.to_thread(vector_db.get_context_for_llm, query=user_input, n_results=2)
            chain = self.prompt_template | self.llm | self.output_parser
            
            response = await chain.ainvoke({
                "role_type": role_type,
                "context": context,
                "chat_history": chat_history,
                "user_input": user_input,
                "audio_conf": "0.0000",
                "video_conf": "0.0000",
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            logger.info(f"纯文本决策完成: 诈骗={response['is_fraud']} | 等级={response['risk_level']}")
            return response
            
        except Exception as e:
            logger.error(f"LLM 纯文本分析异常: {e}", exc_info=True)
            return {
                "is_fraud": False, "risk_level": "safe", "confidence": 0.0,
                "analysis": "大模型调用异常，降级通过", "advice": "系统繁忙"
            }

    async def analyze_multimodal_risk(self, user_input: str, chat_history: str, role_type: str = "青壮年", audio_conf: float = 0.0, video_conf: float = 0.0) -> dict:
        """多模态融合风控分析（带记忆池）"""
        try:
            context = await asyncio.to_thread(vector_db.get_context_for_llm, query=user_input, n_results=2)
            chain = self.prompt_template | self.llm | self.output_parser
            
            response = await chain.ainvoke({
                "role_type": role_type,
                "context": context,
                "chat_history": chat_history,
                "user_input": user_input,
                "audio_conf": f"{audio_conf:.4f}",
                "video_conf": f"{video_conf:.4f}",
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            logger.info(f"多模态决策完成: 诈骗={response['is_fraud']} | 等级={response['risk_level']} | A_conf={audio_conf:.2f} | V_conf={video_conf:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"LLM 多模态融合分析异常: {e}", exc_info=True)
            return {
                "is_fraud": False, "risk_level": "safe", "confidence": 0.0,
                "analysis": "大模型调用异常，降级通过", "advice": "系统繁忙"
            }
    
    async def generate_security_report(self, user: User, recent_calls_with_logs: list) -> str:
        """
        生成个人安全监测报告
        根据用户的画像和近期通话检测记录，生成 Markdown 格式的反诈总结报告
        """
        # 1. 整理用户的近期通话摘要
        call_summary = ""
        risk_count = 0
        
        if not recent_calls_with_logs:
            call_summary = "该用户近期无通话记录。"
        else:
            for item in recent_calls_with_logs:
                # 【修复 1】：正确解包 SQLAlchemy 的 Row 对象 (支持单表和联表两种结果)
                if isinstance(item, tuple) or hasattr(item, '_mapping'):
                    call = item[0]
                    log = item[1] if len(item) > 1 else None
                else:
                    call = item
                    log = None

                # 【修复 2】：安全读取枚举值，防止空值报错
                status = "安全"
                if call.detected_result:
                    val = getattr(call.detected_result, 'name', str(call.detected_result))
                    if 'FAKE' in val:
                        status = "高风险(诈骗)"
                        risk_count += 1
                
                start_time_str = call.start_time.strftime("%Y-%m-%d %H:%M:%S") if call.start_time else "未知时间"
                
                # 提取 AI 检测细节
                details_str = "未检测到异常"
                if log:
                    details_list = []
                    if getattr(log, 'overall_score', 0) > 0:
                        details_list.append(f"综合得分: {log.overall_score:.1f}")
                    if getattr(log, 'detected_keywords', None):
                        details_list.append(f"敏感词: {log.detected_keywords}")
                    if getattr(log, 'voice_confidence', 0) > 0:
                        details_list.append(f"语音伪造率: {log.voice_confidence:.2f}")
                    if getattr(log, 'video_confidence', 0) > 0:
                        details_list.append(f"画面伪造率: {log.video_confidence:.2f}")
                    
                    if details_list:
                        details_str = " | ".join(details_list)

                call_summary += (
                    f"- 时间：{start_time_str}\n"
                    f"  号码：{call.target_name}\n"
                    f"  判定结果：{status}\n"
                    f"  检测详情：{details_str}\n\n"
                )

        # 2. 构建 Prompt
        system_prompt = """你是一个国家级的反诈风控专家。
请根据提供的【用户画像】和【近期通话检测记录】，为该用户生成一份专属的《个人反诈安全监测报告》。

报告要求：
1. 必须使用 Markdown 格式排版，包含标题、加粗、列表等元素，使其美观易读。
2. 结构建议包含：
   -  核心评估结论（综合安全评级：如优秀、良好、极度危险）
   -  近期风险数据回顾（拦截了多少次，触发了哪些敏感词）
   -  暴露的薄弱点分析
   -  专属防骗建议（结合该用户的【角色类型】给出定制化建议）
3. 语气要专业、关怀。"""

        # 【修复 3】：使用 family_id 安全判断监护人绑定状态
        is_bound = '是' if getattr(user, 'family_id', None) else '否'

        user_content = f"""
【用户画像】
- 用户姓名：{user.name or user.username}
- 角色类型：{user.role_type}
- 监护人已绑定：{is_bound}

【近期通话检测记录】(共 {len(recent_calls_with_logs)} 条，其中高风险 {risk_count} 条)
{call_summary}
"""
        
        # 3. 调用大模型
        try:
            # 【修复 4】：直接使用 SystemMessage 和 HumanMessage
            # 彻底避开 ChatPromptTemplate 遇到敏感词 JSON 括号 {} 时引发的 KeyError 崩溃！
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return "## 报告生成失败\n系统当前繁忙，无法生成安全报告，请稍后再试。"

llm_service = LLMService()