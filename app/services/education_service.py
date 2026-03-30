from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc, select
from typing import List, Dict, Any
import asyncio

from app.services.vector_db_service import vector_db
from app.services.llm_service import llm_service
from app.models.user import User
from app.models.call_record import CallRecord
from app.models.education import KnowledgeItem, UserLearningRecord
from app.core.logger import get_logger

logger = get_logger(__name__)

# 用户画像 → 易受骗类型映射
ROLE_VULNERABILITY_MAP = {
    "老人": ["虚假投资理财诈骗", "冒充公检法诈骗", "消除不良记录诈骗", "冒充客服诈骗"],
    "学生": ["刷单返利诈骗", "游戏产品虚假交易", "冒充客服诈骗", "虚假贷款诈骗"],
    "儿童": ["游戏产品虚假交易", "冒充客服诈骗"],
    "宝妈": ["刷单返利诈骗", "虚假投资理财诈骗", "冒充客服诈骗"],
    "青壮年": ["杀猪盘网恋诈骗", "虚假贷款诈骗", "婚恋交友诈骗", "冒充领导熟人诈骗"],
}

# 10种标准诈骗类型
FRAUD_TYPES = [
    "刷单返利诈骗",
    "虚假投资理财诈骗", 
    "冒充客服诈骗",
    "冒充公检法诈骗",
    "杀猪盘网恋诈骗",
    "虚假贷款诈骗",
    "冒充领导熟人诈骗",
    "游戏产品虚假交易",
    "婚恋交友诈骗",
    "消除不良记录诈骗"
]

class EducationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        # 引入向量数据库单例
        self.vector_db = vector_db 

    def _safe_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """统一将向量库返回的 metadata 转成普通 dict。"""
        raw_metadata = item.get("metadata") or {}
        return dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

    def _format_case_item(self, item: Dict[str, Any], default_fraud_type: str = "") -> Dict[str, Any]:
        metadata = self._safe_metadata(item)
        fraud_type = item.get("fraud_type") or metadata.get("fraud_type") or default_fraud_type or "其他"
        return {
            "id": str(item.get("id", "")),
            "title": metadata.get("title") or item.get("title") or f"{fraud_type}案例",
            "content": item.get("content", ""),
            "fraud_type": fraud_type,
            "risk_level": metadata.get("risk_level") or item.get("risk_level"),
            "similarity": item.get("similarity"),
        }

    def _format_slogan_item(self, item: Dict[str, Any], default_fraud_type: str = "") -> Dict[str, Any]:
        metadata = self._safe_metadata(item)
        return {
            "id": str(item.get("id", "")),
            "content": item.get("content", ""),
            "fraud_type": item.get("fraud_type") or metadata.get("fraud_type") or default_fraud_type or "其他",
        }

    def _format_video_item(self, item: Dict[str, Any], default_fraud_type: str = "") -> Dict[str, Any]:
        metadata = self._safe_metadata(item)
        fraud_type = item.get("fraud_type") or metadata.get("fraud_type") or default_fraud_type or "其他"
        return {
            "id": str(item.get("id", "")),
            "title": metadata.get("title") or item.get("title") or f"{fraud_type}视频",
            "url": metadata.get("url") or item.get("url") or "",
            "fraud_type": fraud_type,
            "description": metadata.get("description") or item.get("description") or item.get("content", ""),
        }

    def _format_law_item(self, item: Dict[str, Any], default_fraud_type: str = "") -> Dict[str, Any]:
        metadata = self._safe_metadata(item)
        return {
            "id": str(item.get("id", "")),
            "title": metadata.get("title") or item.get("title") or "反诈法条",
            "content": item.get("content", ""),
            "fraud_type": item.get("fraud_type") or metadata.get("fraud_type") or default_fraud_type or "其他",
            "law_type": metadata.get("law_type") or item.get("law_type") or "反电信网络诈骗法",
            "source_label": item.get("source_label") or metadata.get("source") or "反诈法",
        }

    async def get_personalized_recommendations(self, user_id: int, limit: int = 5):
        """
        功能1: 根据用户画像和近期通话，为用户定制个人学习视频和案例教育
        """
        # 使用异步查询
        result = await self.db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return []

        # 1. 查找近期高危通话记录 (最近3条)
        result = await self.db.execute(
            select(CallRecord).filter(
                CallRecord.user_id == user_id, 
                CallRecord.risk_level.in_(['high', 'critical'])
            ).order_by(desc(CallRecord.created_at)).limit(3)
        )
        recent_calls = result.scalars().all()

        vulnerable_tags = set()
        
        # 提取通话中识别出的诈骗类型
        for call in recent_calls:
            if call.fraud_type:
                vulnerable_tags.add(call.fraud_type)
        
        # 2. 如果近期没有被骗记录，则根据【用户画像 (role_type)】推测他容易受骗的类型
        if not vulnerable_tags and user.role_type:
            # 简单的规则引擎映射（这里对应你之前设置的标准标签池）
            role_mapping = {
                "老人": ["虚假投资理财诈骗", "冒充公检法诈骗", "消除不良记录诈骗"],
                "学生": ["刷单返利诈骗", "游戏产品虚假交易", "冒充客服诈骗"],
                "宝妈": ["刷单返利诈骗", "虚假投资理财诈骗"],
                "青壮年": ["“杀猪盘”网恋诈骗", "虚假贷款诈骗", "婚恋交友诈骗"]
            }
            # 如果能匹配到角色，加入推测标签
            for tag in role_mapping.get(user.role_type, ["刷单返利诈骗"]):
                vulnerable_tags.add(tag)

        # 3. 根据提取到的标签，去 MySQL 中查询对应的学习资料 (视频/图文)
        result = await self.db.execute(
            select(KnowledgeItem).filter(
                KnowledgeItem.fraud_type.in_(vulnerable_tags)
            ).limit(limit)
        )
        recommendations = result.scalars().all()
        
        # 4. 如果没查到针对性的，就返回通用的最新资料兜底
        if not recommendations:
            result = await self.db.execute(
                select(KnowledgeItem).order_by(desc(KnowledgeItem.created_at)).limit(limit)
            )
            recommendations = result.scalars().all()
            
        return recommendations

    async def match_similar_cases_for_call(self, transcript: str, top_k: int = 1):
        """
        功能2: 在向量数据库中匹配相似案例或法律法规
        当用户正在遭遇诈骗通话时调用
        
        案例和法律分开检索，提高检索效率和准确性
        """
        # 从案例库拉取相似案例（anti_fraud_cases）
        similar_cases = self.vector_db.search_similar(
            collection_name="anti_fraud_cases",
            text=transcript,
            top_k=top_k
        )
        
        # 从法律法规库拉取相关法条（anti_fraud_laws）
        similar_laws = self.vector_db.search_similar(
            collection_name="anti_fraud_laws",
            text=transcript,
            top_k=top_k
        )
        
        return {
            "cases": similar_cases,
            "laws": similar_laws
        }

    async def record_user_learning(self, user_id: int, item_id: int, is_completed: bool = False):
        """记录用户的学习进度"""
        result = await self.db.execute(
            select(UserLearningRecord).filter_by(user_id=user_id, item_id=item_id)
        )
        record = result.scalar_one_or_none()
        if not record:
            record = UserLearningRecord(user_id=user_id, item_id=item_id, is_completed=is_completed)
            self.db.add(record)
        else:
            record.is_completed = is_completed
        
        await self.db.commit()
        return record

    # ================= 新增推荐方法 =================
    
    async def recommend_by_user_profile(self, user_id: int, limit: int = 5) -> Dict[str, Any]:
        """
        基于用户画像的个性化推荐
        
        双路召回：
        1. 规则匹配：根据用户角色类型匹配易受骗类型
        2. 向量检索：从对应类型中检索相关内容
        
        Returns:
            {
                "cases": [...],
                "slogans": [...],
                "videos": [...],
                "vulnerability_analysis": "用户易受骗分析",
                "recommended_types": ["诈骗类型1", "诈骗类型2"]
            }
        """
        result = await self.db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return {"error": "用户不存在"}
        
        # 1. 确定用户易受骗类型
        vulnerable_types = []
        if user.role_type and user.role_type in ROLE_VULNERABILITY_MAP:
            vulnerable_types = ROLE_VULNERABILITY_MAP[user.role_type]
        else:
            # 默认推荐通用类型
            vulnerable_types = ["冒充客服诈骗", "刷单返利诈骗"]
        
        logger.info(f"用户 {user_id} 角色: {user.role_type}, 易受骗类型: {vulnerable_types}")
        
        # 2. 为每种易受骗类型检索案例、标语、视频
        all_cases = []
        all_slogans = []
        all_videos = []
        
        for fraud_type in vulnerable_types[:3]:  # 最多取前3种类型
            # 检索案例
            cases = self.vector_db.search_by_fraud_type("anti_fraud_cases", fraud_type, top_k=2)
            all_cases.extend([
                self._format_case_item(case, fraud_type) for case in cases
            ])
            
            # 检索标语
            slogans = self.vector_db.search_by_fraud_type("anti_fraud_slogans", fraud_type, top_k=1)
            all_slogans.extend([
                self._format_slogan_item(slogan, fraud_type) for slogan in slogans
            ])
            
            # 检索视频
            videos = self.vector_db.search_by_fraud_type("anti_fraud_videos", fraud_type, top_k=1)
            all_videos.extend([
                self._format_video_item(video, fraud_type) for video in videos
            ])
        
        # 3. 生成用户易受骗分析
        vulnerability_analysis = await self._generate_vulnerability_analysis(
            user.role_type, vulnerable_types, user.profession, user.marital_status
        )
        
        return {
            "cases": all_cases[:limit],
            "slogans": all_slogans[:limit],
            "videos": all_videos[:limit],
            "vulnerability_analysis": vulnerability_analysis,
            "recommended_types": vulnerable_types
        }
    
    async def recommend_by_conversation(
        self, 
        user_id: int, 
        conversation_text: str, 
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        基于实时对话内容的推荐
        
        当用户正在遭遇诈骗通话时，根据对话内容实时推荐相似案例
        
        Args:
            user_id: 用户ID
            conversation_text: 对话内容/转录文本
            top_k: 返回结果数量
            
        Returns:
            {
                "cases": [...],
                "slogans": [...],
                "similarity_analysis": "相似度分析",
                "alert_message": "预警提示"
            }
        """
        # 1. 向量检索相似案例
        similar_cases = self.vector_db.search_similar(
            collection_name="anti_fraud_cases",
            text=conversation_text,
            top_k=top_k
        )
        
        # 2. 检索相关标语
        similar_slogans = self.vector_db.search_similar(
            collection_name="anti_fraud_slogans",
            text=conversation_text,
            top_k=2
        )
        
        # 3. 获取用户画像进行个性化
        result = await self.db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalar_one_or_none()
        user_profile = user.role_type if user else "未知"
        
        # 4. 生成相似度分析和预警提示
        analysis_result = await self._analyze_conversation_similarity(
            conversation_text, similar_cases, user_profile
        )
        formatted_cases = [self._format_case_item(case) for case in similar_cases]
        formatted_slogans = [self._format_slogan_item(slogan) for slogan in similar_slogans]
        
        return {
            "cases": formatted_cases,
            "slogans": formatted_slogans,
            "similarity_analysis": analysis_result.get("analysis", ""),
            "alert_message": analysis_result.get("alert", ""),
            "matched_fraud_types": list(set([
                case.get("fraud_type", "")
                for case in formatted_cases
                if case.get("fraud_type") and (case.get("similarity") or 0) > 0.7
            ]))
        }
    
    async def recommend_from_case_and_law_library(
        self,
        user_id: int,
        fraud_type: str = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        从案例库和法律库中推荐内容
        
        融合大赛提供的案例库和法律库数据
        案例和法律分别存储在不同的集合中，分开检索
        
        Args:
            user_id: 用户ID
            fraud_type: 指定诈骗类型（可选）
            limit: 返回数量
            
        Returns:
            {
                "cases": [案例列表],
                "laws": [法律条文列表],
                "mixed_recommendations": [混合推荐],
                "source_stats": {"cases": n, "laws": n}
            }
        """
        result = await self.db.execute(select(User).filter(User.user_id == user_id))
        user = result.scalar_one_or_none()
        
        # 1. 确定查询的诈骗类型
        query_types = []
        if fraud_type:
            query_types = [fraud_type]
        elif user and user.role_type:
            query_types = ROLE_VULNERABILITY_MAP.get(user.role_type, ["刷单返利诈骗"])
        else:
            query_types = ["刷单返利诈骗", "冒充客服诈骗"]
        
        all_cases = []
        all_laws = []
        
        # 2. 从案例集合检索案例（anti_fraud_cases）
        for ftype in query_types[:2]:  # 最多2种类型
            cases = self.vector_db.search_by_fraud_type("anti_fraud_cases", ftype, top_k=limit)
            for case in cases:
                formatted_case = self._format_case_item(case, ftype)
                formatted_case["source_label"] = "大赛案例库"
                all_cases.append(formatted_case)
        
        # 3. 从法律集合检索法律条文（anti_fraud_laws）
        for ftype in query_types[:2]:
            laws = self.vector_db.search_by_fraud_type("anti_fraud_laws", ftype, top_k=2)
            for law in laws:
                formatted_law = self._format_law_item(law, ftype)
                formatted_law["source_label"] = "反诈法"
                all_laws.append(formatted_law)
        
        # 4. 混合推荐：案例 + 法律条文
        mixed = []
        case_idx = 0
        law_idx = 0
        
        while len(mixed) < limit and (case_idx < len(all_cases) or law_idx < len(all_laws)):
            if case_idx < len(all_cases):
                mixed.append({
                    "type": "case",
                    "content": all_cases[case_idx],
                    "priority": 1
                })
                case_idx += 1
            if law_idx < len(all_laws) and len(mixed) < limit:
                mixed.append({
                    "type": "law",
                    "content": all_laws[law_idx],
                    "priority": 2
                })
                law_idx += 1
        
        return {
            "cases": all_cases[:limit],
            "laws": all_laws[:limit//2],
            "mixed_recommendations": mixed,
            "source_stats": {
                "cases": len(all_cases),
                "laws": len(all_laws)
            },
            "query_types": query_types
        }
    
    async def _generate_vulnerability_analysis(
        self, 
        role_type: str, 
        vulnerable_types: List[str],
        profession: str = None,
        marital_status: str = None
    ) -> str:
        """
        生成用户易受骗分析文本
        """
        if not role_type:
            return "暂无用户画像信息，建议完善个人资料以获取精准推荐。"
        
        # 构建分析文本
        analysis_parts = [f"根据您的角色类型【{role_type}】，系统分析您可能面临以下诈骗风险："]
        
        type_descriptions = {
            "刷单返利诈骗": "骗子以'兼职刷单、高额返利'为诱饵，先让您垫付小额资金，再要求大额投入。",
            "虚假投资理财诈骗": "冒充投资专家，诱导您下载虚假投资APP，前期小额盈利，后期无法提现。",
            "冒充客服诈骗": "冒充电商、快递客服，以'退款、理赔'为由诱导您提供银行卡信息或下载屏幕共享软件。",
            "冒充公检法诈骗": "冒充公安、检察院，以'涉嫌犯罪'为由恐吓您将资金转入'安全账户'。",
            "杀猪盘网恋诈骗": "在社交软件培养感情后，诱导您投资博彩或虚假理财平台。",
            "虚假贷款诈骗": "以'无抵押、低利息'为诱饵，要求您先交手续费、保证金。",
            "冒充领导熟人诈骗": "冒充您的领导或亲友，以'急用钱'为由要求转账。",
            "游戏产品虚假交易": "以低价出售游戏装备或账号为由，诱导您在虚假平台交易。",
            "婚恋交友诈骗": "以婚恋为名，逐渐诱导您投资或借钱。",
            "消除不良记录诈骗": "声称可以消除征信不良记录，要求您支付费用。"
        }
        
        for i, fraud_type in enumerate(vulnerable_types[:3], 1):
            desc = type_descriptions.get(fraud_type, "请提高警惕，谨防此类诈骗。")
            analysis_parts.append(f"{i}. 【{fraud_type}】{desc}")
        
        analysis_parts.append("\n建议您学习以上相关案例，提高防范意识。")
        
        return "\n\n".join(analysis_parts)
    
    async def _analyze_conversation_similarity(
        self, 
        conversation: str, 
        similar_cases: List[Dict],
        user_profile: str
    ) -> Dict[str, str]:
        """
        分析对话与相似案例的关联，生成预警提示
        """
        if not similar_cases:
            return {
                "analysis": "暂未匹配到相似案例。",
                "alert": ""
            }
        
        # 获取最高相似度
        max_similarity = max([case.get("similarity", 0) for case in similar_cases], default=0)
        
        if max_similarity > 0.8:
            top_case = similar_cases[0]
            fraud_type = top_case.get("metadata", {}).get("fraud_type", "未知类型")
            return {
                "analysis": f"当前对话与【{fraud_type}】案例高度相似（相似度{max_similarity:.1%}），存在极高诈骗风险！",
                "alert": f"⚠️ 警告：检测到您可能正在遭遇{fraud_type}！对方话术与已知诈骗案例高度吻合，请立即挂断电话！"
            }
        elif max_similarity > 0.6:
            top_case = similar_cases[0]
            fraud_type = top_case.get("metadata", {}).get("fraud_type", "未知类型")
            return {
                "analysis": f"当前对话与【{fraud_type}】案例较为相似（相似度{max_similarity:.1%}），请提高警惕。",
                "alert": f"⚠️ 提示：当前通话内容与{fraud_type}案例有相似之处，请注意核实对方身份，不要转账！"
            }
        else:
            return {
                "analysis": f"当前对话与已知案例相似度较低（{max_similarity:.1%}），但仍需保持警惕。",
                "alert": ""
            }