"""
MDP 决策事件模型
用于持久化 state -> action -> next_state -> reward 轨迹
"""
from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, Text, String, Boolean
from sqlalchemy.sql import func

from app.db.database import Base


class MDPDecisionEvent(Base):
    __tablename__ = "mdp_decision_events"

    event_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    call_id = Column(Integer, ForeignKey("call_records.call_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False, default=1, comment="同一通话内的决策步序号")
    trigger_type = Column(String(30), default="text", comment="触发来源: text/audio/video/image")

    state_key = Column(String(255), nullable=False, index=True, comment="当前离散状态唯一键")
    state_json = Column(Text, nullable=False, comment="当前状态完整 JSON")
    next_state_key = Column(String(255), nullable=True, index=True, comment="下一状态唯一键")
    next_state_json = Column(Text, nullable=True, comment="下一状态完整 JSON")

    fused_score = Column(Float, nullable=False, default=0.0)
    text_conf = Column(Float, nullable=False, default=0.0)
    audio_conf = Column(Float, nullable=False, default=0.0)
    video_conf = Column(Float, nullable=False, default=0.0)

    llm_result_json = Column(Text, nullable=True, comment="LLM 分类结果 JSON")
    rule_hit_json = Column(Text, nullable=True, comment="命中规则详情 JSON")

    action_level = Column(Integer, nullable=False, comment="0/1/2 -> LEVEL_1/2/3")
    defense_payload_json = Column(Text, nullable=True, comment="下发到前端/通知层的动作配置")

    reward = Column(Float, nullable=True, comment="轨迹奖励")
    reward_source = Column(String(30), nullable=True, comment="reward 来源: transition/final/manual")
    label_status = Column(String(20), default="open", comment="open/resolved/finalized")

    policy_version = Column(String(100), nullable=True, comment="使用的策略版本")
    reason_codes_json = Column(Text, nullable=True, comment="动作解释码 JSON")
    fallback_used = Column(Boolean, default=False, comment="是否使用规则回退")

    ai_detection_log_id = Column(Integer, ForeignKey("ai_detection_logs.log_id", ondelete="SET NULL"), nullable=True)
    message_log_id = Column(Integer, ForeignKey("message_logs.id", ondelete="SET NULL"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), comment="创建时间")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), comment="更新时间")

    def __repr__(self):
        return f"<MDPDecisionEvent(event_id={self.event_id}, call_id={self.call_id}, step={self.step_index}, action={self.action_level})>"
