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
from app.services.long_term_memory_service import long_term_memory_service
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger(__name__)

# 定义大模型强制输出的 JSON 结构
class RiskAssessmentOutput(BaseModel):
    is_fraud: bool = Field(description="是否判定为诈骗风险")
    risk_level: str = Field(description="风险等级分类：'safe' (安全), 'suspicious' (可疑), 'fake' (高危诈骗)")
    match_script: str = Field(description="匹配的典型诈骗剧本，如'杀猪盘','冒充公检法','刷单返利','虚假投资','亲属出事'等。如未匹配填'无'")
    fraud_type: str = Field(description="诈骗类型分类，必须从以下10种标准类型中选择：刷单返利诈骗、虚假投资理财诈骗、冒充客服诈骗、冒充公检法诈骗、杀猪盘网恋诈骗、虚假贷款诈骗、冒充领导熟人诈骗、游戏产品虚假交易、婚恋交友诈骗、消除不良记录诈骗。如无法确定填'其他'")
    intent: str = Field(description="核心意图识别，如'诱导转账/汇款','索要验证码/密码','要求屏幕共享','常规业务沟通'等")
    analysis: str = Field(description="对用户意图和话术套路的逻辑分析过程")
    advice: str = Field(description="针对特定受众人群给出的个性化防骗建议")

class LLMService:
    def __init__(self):
        self.output_parser = JsonOutputParser(pydantic_object=RiskAssessmentOutput)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的多模态反诈智能体助手，具备强大的意图识别与复杂逻辑推理能力。
你的核心任务是：综合分析上下文与多模态特征，提取对方核心意图，并分类出具体的诈骗剧本。

【当前用户的综合画像】：
{user_profile}
【当前通话场景】：
{call_type}

注意：请严格根据用户的综合画像动态调整你的审查重点。以下是不同特征人群的易受骗场景：
- 老年人：对推销特效药、高息理财、冒充公检法恐吓极度敏感。
- 宝妈/全职：极易遇到“兼职刷单”、“买返利金”、“免费领母婴用品”诈骗。
- 单身/离异：极易遇到“杀猪盘”（网恋诱导博彩/投资诈骗）。
- 公司财务/出纳：极易遇到“冒充老板/公检法要求紧急对公转账”。
- 学生/儿童：对免费领皮肤、解除防沉迷、注销校园贷、非官方低价票高度警惕。

【十大典型诈骗剧本库】（请根据实际情况选择，参考以下类别）：
1. 刷单返利诈骗 2. 虚假投资理财诈骗 3. 冒充客服诈骗 4. 冒充公检法诈骗 5. “杀猪盘”网恋诈骗
6. 虚假贷款诈骗 7. 冒充领导熟人诈骗 8. 游戏产品虚假交易 9. 婚恋交友诈骗 10. 消除不良记录诈骗

【知识库参考案例（RAG检索结果）】：
{context}

【近期对话上下文】（请结合这些历史记录来理解当前输入的真实意图）：
{chat_history}

【用户长期记忆（跨通话历史）】：
{long_term_memory}

【底层AI模型多模态特征分析结果】（极其重要！）：
- 本地文本预检(Text-ONNX)置信度：{text_conf} （0~1.0，>0.85通常代表话术高度匹配已知模板）
- 语音伪造(Voice Clone)置信度：{audio_conf} （0~1.0，>0.8通常代表极大可能是AI合成变声）
- 视频伪造(Deepfake)置信度：{video_conf} （0~1.0，>0.75通常代表画面经过AI换脸或唇形篡改）

【多模态融合推理规则】：
1. 意图提取与连贯性：骗子通常会将“我是领导”、“出事了”、“快打钱”分段发送。结合【最新输入】与【上下文】，准确提取对方的真实目的（intent）。
2. 剧本匹配与交叉验证：如果文本涉及要钱/转账，且【语音/视频伪造置信度】高，必须判定 is_fraud=True，risk_level="fake"，并在 match_script 中给出对应的剧本名称。
3. 纠正误报：如果语音置信度较高，但结合上下文完全是正常业务（intent为常规沟通，如送件），请结合常识判定为 safe，纠正底层误报。
4. 双重印证：如果【本地文本预检置信度】极高(>0.9)，即使音视频真实，也说明内容极具煽动性，请重点提取其危险意图并倾向判定为高风险。
5. 历史记忆参考：如果用户有多次忽略告警的记录，应提高告警强度；如果用户曾遭遇相似诈骗，应提前预警。

请综合推理，并严格按照 JSON 格式输出最终裁定结果。
{format_instructions}

【极其重要】：你的输出必须且只能是一个合法的 JSON 对象。绝对不能包含任何 Markdown 语法（如 ```json ），不要包含任何解释性文本、不要有任何前言或后语。必须以大括号 {{ 开始，以大括号 }} 结束。"""),
            ("human", "【当前最新输入(含语音转写)】：{user_input}")
        ])

    def _create_llm(self) -> ChatOpenAI:
        """
        为每次调用创建独立 LLM 客户端，避免 Celery 线程池下复用异步客户端导致
        `Event loop is closed`。
        """
        return ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0.1,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL
        )

    def _parse_json_response(self, content: str) -> dict:
        """清理并解析模型返回的 JSON 内容。"""
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return self.output_parser.parse(cleaned.strip())


    async def analyze_text_risk(self, user_input: str, chat_history: str, user_profile: str = "青壮年") -> dict:
        """带记忆池文本风控分析"""
        try:
            context = await asyncio.to_thread(vector_db.get_context_for_llm, query=user_input, n_results=2)
            llm = self._create_llm()
            chain = self.prompt_template | llm | self.output_parser
            
            response = await chain.ainvoke({
                "user_profile": user_profile,
                "call_type": "普通文本对话",
                "context": context,
                "chat_history": chat_history,
                "user_input": user_input,
                "audio_conf": "0.0000",
                "text_conf": "0.0000",
                "video_conf": "0.0000",
                "format_instructions": self.output_parser.get_format_instructions()
            })
            
            logger.info(f"纯文本决策完成: 诈骗={response['is_fraud']} | 等级={response['risk_level']}")
            return response
            
        except Exception as e:
            logger.error(f"LLM 纯文本分析异常: {e}", exc_info=True)
            return {
                "is_fraud": False, "risk_level": "safe", 
                "match_script": "无", "intent": "无法识别",
                "analysis": "大模型调用异常，降级通过", "advice": "系统繁忙"
            }

    def _get_scene_specific_prompt(self, call_type: str) -> str:
        """
        根据通话场景获取特定的检测提示
        
        Args:
            call_type: 通话类型/场景描述
            
        Returns:
            场景特定的提示词
        """
        call_type_lower = call_type.lower()
        
        # 纯文字聊天场景
        if "文字" in call_type or "文本" in call_type or "qq消息" in call_type_lower or "微信消息" in call_type_lower:
            return """
【文字聊天场景专项检测要点】：
1. 聊天双方身份识别：
   - 仔细分析对话中双方的身份标识（昵称、头像位置、消息气泡颜色）
   - 识别哪一方是用户（被保护对象），哪一方是陌生人/可疑对象
   - 在分析中明确指出可疑消息的发送者身份

2. 文字聊天特有风险：
   - 重点关注话术套路：杀猪盘培养感情、刷单返利诱导、冒充客服退款
   - 识别诱导性语言："点击链接"、"下载APP"、"先垫付小额"
   - 注意紧急性话术："限时"、"名额有限"、"马上截止"
   - 警惕信息收集：索要身份证号、银行卡、验证码、密码

3. 对话上下文分析：
   - 分析多轮对话中的意图演变（从正常聊天转向诱导）
   - 识别"钓鱼"过程：先建立信任，后实施诈骗
   - 标注可疑消息的发送方和接收方
"""
        
        # 语音聊天场景（QQ/微信语音）
        elif "语音" in call_type or "语音聊天" in call_type:
            return """
【语音聊天场景专项检测要点】：
1. 语音伪造识别：
   - 结合音频置信度判断是否为AI合成语音
   - 注意变声特征：声音不自然、背景杂音异常、语速机械
   - 警惕"熟人"声音：可能是语音克隆技术伪造

2. 语音聊天特有风险：
   - 紧急性诱导："马上转账"、"来不及解释了"
   - 情感操控：利用亲情、友情、爱情建立信任
   - 信息确认：语音中索要敏感信息（验证码、密码）

3. 多模态融合：
   - 结合文字转写内容分析语音意图
   - 音频置信度高 + 诱导性话术 = 高风险
"""
        
        # 电话通话场景
        elif "电话" in call_type or "phone" in call_type_lower:
            return """
【电话通话场景专项检测要点】：
1. 来电身份识别：
   - 注意陌生号码、境外号码、虚拟运营商号码
   - 警惕"官方"来电：公检法、银行、客服、快递

2. 电话诈骗特征：
   - 冒充公检法："涉嫌洗钱"、"账户冻结"、"配合调查"
   - 冒充客服："商品质量问题"、"退款"、"理赔"
   - 冒充熟人："我是你领导"、"换号了"、"急需用钱"

3. 语音伪造检测：
   - 结合音频置信度判断声音真实性
   - 注意机械音、回声异常、背景噪声不自然
"""
        
        # 视频通话场景
        elif "视频" in call_type or "video" in call_type_lower:
            return """
【视频通话场景专项检测要点】：
1. 深度伪造识别：
   - 结合视频置信度判断是否为AI换脸
   - 注意面部不自然：眼神呆滞、表情僵硬、边缘模糊
   - 唇形同步异常：说话与口型不匹配

2. 视频通话特有风险：
   - 屏幕共享诱导："指导操作"、"查看账户"
   - 实时换脸：冒充熟人进行视频通话
   - 环境伪造：虚假背景、预录视频

3. 三模态融合：
   - 视频异常 + 音频异常 + 诱导话术 = 极高风险
   - 任何单一模态高度可疑都应引起重视
"""
        
        # 默认场景
        else:
            return """
【通用检测要点】：
1. 综合分析多模态特征（文本、音频、视频）
2. 识别常见诈骗剧本和话术套路
3. 结合用户画像判断易受骗场景
4. 注意意图的连贯性和演变过程
"""

    async def analyze_multimodal_risk(self, user_input: str, chat_history: str, user_profile: str = "青壮年", 
        call_type: str = "普通通话", audio_conf: float = 0.0, 
        video_conf: str = "0.0", text_conf: float = 0.0, 
        db = None, user_id: int = None,
        chat_speakers: dict = None) -> dict:
        """
        多模态融合风控分析（增强版：支持长期记忆、场景自适应）
        
        Args:
            user_input: 用户输入文本
            chat_history: 对话历史
            user_profile: 用户画像
            call_type: 通话类型
            audio_conf: 音频置信度
            video_conf: 视频置信度
            text_conf: 文本置信度
            db: 数据库会话
            user_id: 用户ID（用于获取长期记忆）
            chat_speakers: 聊天双方信息 {"user": "用户昵称", "other": "对方昵称", "messages": [{"sender": "", "content": ""}]}
        """
        try:
            # 1. 获取知识库上下文 (RAG)
            context = await asyncio.to_thread(vector_db.get_context_for_llm, query=user_input, n_results=2)
            
            # 2. 获取长期记忆
            long_term_memory = ""
            if db and user_id:
                try:
                    long_term_memory = await long_term_memory_service.get_context_for_prompt(db, user_id)
                except Exception as mem_e:
                    logger.warning(f"获取长期记忆失败: {mem_e}")
            
            # 3. 如果提供了db和user_id，获取实时推荐案例
            recommendations = None
            if db and user_id:
                try:
                    from app.services.education_service import EducationService
                    edu_service = EducationService(db)
                    recommendations = await edu_service.recommend_by_conversation(
                        user_id=user_id,
                        conversation_text=user_input,
                        top_k=2
                    )
                    if recommendations.get("cases"):
                        rec_context = "\n\n【系统推荐的相似案例】:\n"
                        for i, case in enumerate(recommendations["cases"][:2], 1):
                            rec_context += f"案例{i}: {case.get('title', '')}\n"
                            rec_context += f"类型: {case.get('fraud_type', '')}\n"
                            rec_context += f"内容: {case.get('content', '')[:200]}...\n\n"
                        context += rec_context
                except Exception as rec_e:
                    logger.warning(f"获取推荐案例失败: {rec_e}")
            
            # 4. 获取场景特定提示
            scene_prompt = self._get_scene_specific_prompt(call_type)
            
            # 5. 构建聊天双方信息
            speaker_info = ""
            if chat_speakers:
                speaker_info = f"""
【聊天双方身份信息】：
- 用户（被保护对象）：{chat_speakers.get('user', '未知')}
- 对方（陌生人/可疑对象）：{chat_speakers.get('other', '未知')}

【对话消息列表】（带发送者标识）：
"""
                for msg in chat_speakers.get('messages', []):
                    sender = msg.get('sender', '未知')
                    content = msg.get('content', '')
                    speaker_info += f"[{sender}]: {content}\n"
            
            # 6. 构建动态提示词
            dynamic_system_prompt = f"""你是一个专业的多模态反诈智能体助手，具备强大的意图识别与复杂逻辑推理能力。
你的核心任务是：综合分析上下文与多模态特征，提取对方核心意图，并分类出具体的诈骗剧本。

【当前用户的综合画像】：
{user_profile}
【当前通话场景】：
{call_type}

{scene_prompt}

{speaker_info}

注意：请严格根据用户的综合画像动态调整你的审查重点。以下是不同特征人群的易受骗场景：
- 老年人：对推销特效药、高息理财、冒充公检法恐吓极度敏感。
- 宝妈/全职：极易遇到"兼职刷单"、"买返利金"、"免费领母婴用品"诈骗。
- 单身/离异：极易遇到"杀猪盘"（网恋诱导博彩/投资诈骗）。
- 公司财务/出纳：极易遇到"冒充老板/公检法要求紧急对公转账"。
- 学生/儿童：对免费领皮肤、解除防沉迷、注销校园贷、非官方低价票高度警惕。

【十大典型诈骗剧本库】（请根据实际情况选择，参考以下类别）：
1. 刷单返利诈骗 2. 虚假投资理财诈骗 3. 冒充客服诈骗 4. 冒充公检法诈骗 5. "杀猪盘"网恋诈骗
6. 虚假贷款诈骗 7. 冒充领导熟人诈骗 8. 游戏产品虚假交易 9. 婚恋交友诈骗 10. 消除不良记录诈骗

【知识库参考案例（RAG检索结果）】：
{context}

【近期对话上下文】（请结合这些历史记录来理解当前输入的真实意图）：
{chat_history}

【用户长期记忆（跨通话历史）】：
{long_term_memory or "该用户暂无历史记忆。"}

【底层AI模型多模态特征分析结果】（极其重要！）：
- 本地文本预检(Text-ONNX)置信度：{text_conf:.4f} （0~1.0，>0.85通常代表话术高度匹配已知模板）
- 语音伪造(Voice Clone)置信度：{audio_conf:.4f} （0~1.0，>0.8通常代表极大可能是AI合成变声）
- 视频伪造(Deepfake)置信度：{video_conf} （0~1.0，>0.75通常代表画面经过AI换脸或唇形篡改）

【多模态融合推理规则】：
1. 意图提取与连贯性：骗子通常会将"我是领导"、"出事了"、"快打钱"分段发送。结合【最新输入】与【上下文】，准确提取对方的真实目的（intent）。
2. 剧本匹配与交叉验证：如果文本涉及要钱/转账，且【语音/视频伪造置信度】高，必须判定 is_fraud=True，risk_level="fake"，并在 match_script 中给出对应的剧本名称。
3. 纠正误报：如果语音置信度较高，但结合上下文完全是正常业务（intent为常规沟通，如送件），请结合常识判定为 safe，纠正底层误报。
4. 双重印证：如果【本地文本预检置信度】极高(>0.9)，即使音视频真实，也说明内容极具煽动性，请重点提取其危险意图并倾向判定为高风险。
5. 历史记忆参考：如果用户有多次忽略告警的记录，应提高告警强度；如果用户曾遭遇相似诈骗，应提前预警。
6. 聊天双方识别：在文字聊天场景中，明确指出可疑消息的发送者身份（是用户发送的还是对方发送的）。

请综合推理，并严格按照以下 JSON 结构输出最终裁定结果：
{{
  "is_fraud": true,
  "risk_level": "safe|suspicious|fake",
  "match_script": "匹配的诈骗剧本，没有则填 无",
  "fraud_type": "诈骗类型，没有则填 其他",
  "intent": "核心意图识别",
  "analysis": "逻辑分析过程",
  "advice": "给用户的防骗建议"
}}

【极其重要】：
1. 你的输出必须且只能是一个合法 JSON 对象。
2. 不要包含 Markdown 代码块，不要附加解释性文本。
3. `risk_level` 只能取 `safe`、`suspicious`、`fake` 三个值之一。"""

            llm = self._create_llm()
            messages = [
                SystemMessage(content=dynamic_system_prompt),
                HumanMessage(content=f"【当前最新输入(含语音转写)】：{user_input}")
            ]
            raw_response = await llm.ainvoke(messages)
            response = self._parse_json_response(raw_response.content)
            
            # 4. 将推荐信息添加到响应中
            if recommendations:
                response["recommendations"] = {
                    "cases": recommendations.get("cases", []),
                    "slogans": recommendations.get("slogans", []),
                    "alert_message": recommendations.get("alert_message", "")
                }
            
            # 5. 提取并保存长期记忆
            if db and user_id:
                try:
                    # 这里需要call_id，但当前方法没有传入，需要上层调用时处理
                    pass
                except Exception as save_e:
                    logger.warning(f"保存长期记忆失败: {save_e}")
            
            logger.info(f"多模态决策完成: 诈骗={response['is_fraud']} | 等级={response['risk_level']} | T_conf={text_conf:.2f} | A_conf={audio_conf:.2f} | V_conf={video_conf}")
            return response
            
        except Exception as e:
            logger.error(f"LLM 多模态融合分析异常: {e}", exc_info=True)
            return {
                "is_fraud": False, "risk_level": "safe", 
                "match_script": "无", "intent": "无法识别",
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
        system_prompt = """你是一个专业的反诈风控专家。
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
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]

            llm = self._create_llm()
            response = await llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return "## 报告生成失败\n系统当前繁忙，无法生成安全报告，请稍后再试。"
    
    async def generate_security_report_stream(self, user: User, recent_calls_with_logs: list):
        """
        流式生成个人安全监测报告
        返回异步生成器，每次yield一个内容片段
        """
        # 1. 整理用户的近期通话摘要
        call_summary = ""
        risk_count = 0
        
        if not recent_calls_with_logs:
            call_summary = "该用户近期无通话记录。"
        else:
            for item in recent_calls_with_logs:
                if isinstance(item, tuple) or hasattr(item, '_mapping'):
                    call = item[0]
                    log = item[1] if len(item) > 1 else None
                else:
                    call = item
                    log = None

                status = "安全"
                if call.detected_result:
                    val = getattr(call.detected_result, 'name', str(call.detected_result))
                    if 'FAKE' in val:
                        status = "高风险(诈骗)"
                        risk_count += 1
                
                start_time_str = call.start_time.strftime("%Y-%m-%d %H:%M:%S") if call.start_time else "未知时间"
                
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
        system_prompt = """你是一个专业的反诈风控专家。
请根据提供的【用户画像】和【近期通话检测记录】，为该用户生成一份专属的《个人反诈安全监测报告》。

报告要求：
1. 必须使用 Markdown 格式排版，包含标题、加粗、列表等元素，使其美观易读。
2. 结构建议包含：
   -  核心评估结论（综合安全评级：如优秀、良好、极度危险）
   -  近期风险数据回顾（拦截了多少次，触发了哪些敏感词）
   -  暴露的薄弱点分析
   -  专属防骗建议（结合该用户的【角色类型】给出定制化建议）
3. 语气要专业、关怀。"""

        is_bound = '是' if getattr(user, 'family_id', None) else '否'

        user_content = f"""
【用户画像】
- 用户姓名：{user.name or user.username}
- 角色类型：{user.role_type}
- 监护人已绑定：{is_bound}

【近期通话检测记录】(共 {len(recent_calls_with_logs)} 条，其中高风险 {risk_count} 条)
{call_summary}
"""
        
        # 3. 流式调用大模型
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content)
            ]

            llm = self._create_llm()
            # 使用 astream 方法进行流式输出
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Failed to stream security report: {e}")
            yield "## 报告生成失败\n系统当前繁忙，无法生成安全报告，请稍后再试。"
    
    async def generate_final_summary(self, chat_history: list | str) -> dict:
        """
        通话结束时，根据全局对话历史生成最终总结
        """
        # 1. 安全处理历史记录格式（兼容字符串和列表字典）
        if isinstance(chat_history, list):
            try:
                history_text = "\n".join([f"{msg.get('role', 'User')}: {msg.get('content', '')}" for msg in chat_history])
            except Exception:
                history_text = str(chat_history)
        else:
            history_text = str(chat_history)
            
        # 2. 构建符合 LangChain 规范的 Prompt
        system_prompt = """你是一个专业的反诈风控专家。一通电话或聊天刚刚结束，以下是完整的对话记录。
请根据整个对话的上下文，进行一次全局视角的复盘分析。

必须且只能返回合法的 JSON 对象，不要包含 Markdown 标记（如 ```json），字段要求如下：
{
    "risk_level": "风险等级评估 (选项: safe, suspicious, fake, high, critical)",
    "analysis": "通话总结分析 (描述对方的整体意图、使用了什么套路，或者是否属于正常沟通)",
    "advice": "给用户的最终防范建议 (如果安全提示保持警惕；如果有风险给出止损建议)"
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"以下是完整的对话记录：\n{history_text}")
        ]

        try:
            # 3. 直接调用 LLM 并尝试解析返回结果
            llm = self._create_llm()
            response = await llm.ainvoke(messages)
            result = self._parse_json_response(response.content)
            return result
            
        except Exception as e:
            logger.error(f"全局总结LLM调用失败: {e}", exc_info=True)
            return {
                "risk_level": "safe",
                "analysis": "大模型全局复盘分析生成失败，请参考实时检测记录。",
                "advice": "系统暂无额外建议。"
            }
llm_service = LLMService()