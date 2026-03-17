from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.services.vector_db_service import vector_db
from app.models.user import User
from app.models.call_record import CallRecord
from app.models.education import KnowledgeItem, UserLearningRecord

class EducationService:
    def __init__(self, db: Session):
        self.db = db
        # 引入我们刚才写好的向量数据库单例
        self.vector_db = vector_db 

    async def get_personalized_recommendations(self, user_id: int, limit: int = 5):
        """
        功能1: 根据用户画像和近期通话，为用户定制个人学习视频和案例教育
        """
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return []

        # 1. 查找近期高危通话记录 (最近3条)
        recent_calls = self.db.query(CallRecord).filter(
            CallRecord.user_id == user_id, 
            CallRecord.risk_level.in_(['high', 'critical'])
        ).order_by(desc(CallRecord.created_at)).limit(3).all()

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
        recommendations = self.db.query(KnowledgeItem).filter(
            KnowledgeItem.fraud_type.in_(vulnerable_tags)
        ).limit(limit).all()
        
        # 4. 如果没查到针对性的，就返回通用的最新资料兜底
        if not recommendations:
            recommendations = self.db.query(KnowledgeItem).order_by(desc(KnowledgeItem.created_at)).limit(limit).all()
            
        return recommendations

    async def match_similar_cases_for_call(self, transcript: str, top_k: int = 1):
        """
        功能2: 在向量数据库中匹配相似案例或法律法规
        当用户正在遭遇诈骗通话时调用
        """
        # 从防诈案例库拉取相似案例
        similar_cases = self.vector_db.search_similar(
            collection_name="anti_fraud_cases",
            text=transcript,
            top_k=top_k
        )
        
        # 从法律法规库拉取相关法条
        similar_laws = self.vector_db.search_similar(
            collection_name="anti_fraud_laws",
            text=transcript,
            top_k=top_k
        )
        
        return {
            "cases": similar_cases,
            "laws": similar_laws
        }

    def record_user_learning(self, user_id: int, item_id: int, is_completed: bool = False):
        """记录用户的学习进度"""
        record = self.db.query(UserLearningRecord).filter_by(user_id=user_id, item_id=item_id).first()
        if not record:
            record = UserLearningRecord(user_id=user_id, item_id=item_id, is_completed=is_completed)
            self.db.add(record)
        else:
            record.is_completed = is_completed
        
        self.db.commit()
        return record