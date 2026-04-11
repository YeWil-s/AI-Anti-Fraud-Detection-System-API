from __future__ import annotations

from app.models.user import User

from .mdp_types import (
    EnvironmentType,
    InteractionDepth,
    MDPObservation,
    MDPState,
    RiskLevel,
    RuleHitLevel,
    UserVulnerability,
)


class StateBuilder:
    """统一在线推理与离线训练的状态离散化逻辑。"""

    HIGH_RISK_ROLES = ["老人", "儿童", "学生"]
    HIGH_RISK_OCCUPATIONS = ["学生", "退休", "无业"]
    LOW_RISK_OCCUPATIONS = ["程序员", "教师", "公务员", "医生", "it", "teacher"]

    def calculate_vulnerability(self, user: User | None) -> UserVulnerability:
        if not user:
            return UserVulnerability.MEDIUM

        profession = (getattr(user, "profession", "") or "").lower()
        role_type = getattr(user, "role_type", "") or ""
        marital_status = getattr(user, "marital_status", "") or ""

        if any(role in role_type for role in self.HIGH_RISK_ROLES) or any(
            job in profession for job in self.HIGH_RISK_OCCUPATIONS
        ):
            return UserVulnerability.HIGH

        if any(job in profession for job in self.LOW_RISK_OCCUPATIONS) and marital_status == "已婚":
            return UserVulnerability.LOW

        return UserVulnerability.MEDIUM

    def build_risk_level(self, risk_score: float) -> RiskLevel:
        if risk_score >= 80:
            return RiskLevel.HIGH
        if risk_score >= 40:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def build_interaction_depth(self, message_count: int) -> InteractionDepth:
        if message_count > 10:
            return InteractionDepth.DEEP
        if message_count > 3:
            return InteractionDepth.MEDIUM
        return InteractionDepth.SHALLOW

    def build_environment_type(self, environment_type: str) -> EnvironmentType:
        normalized = (environment_type or "").strip().lower()
        mapping = {
            "text_chat": EnvironmentType.TEXT_CHAT,
            "voice_chat": EnvironmentType.VOICE_CHAT,
            "phone_call": EnvironmentType.PHONE_CALL,
            "video_call": EnvironmentType.VIDEO_CALL,
            "unknown": EnvironmentType.UNKNOWN,
        }
        return mapping.get(normalized, EnvironmentType.UNKNOWN)

    def build_rule_hit_level(self, rule_hit_level: int) -> RuleHitLevel:
        if rule_hit_level >= 4:
            return RuleHitLevel.CRITICAL
        if rule_hit_level >= 3:
            return RuleHitLevel.HIGH
        if rule_hit_level >= 2:
            return RuleHitLevel.MEDIUM
        if rule_hit_level >= 1:
            return RuleHitLevel.LOW
        return RuleHitLevel.NONE

    def build_state(self, user: User | None, observation: MDPObservation) -> MDPState:
        return MDPState(
            user_vulnerability=self.calculate_vulnerability(user),
            risk_level=self.build_risk_level(observation.risk_score),
            interaction_depth=self.build_interaction_depth(observation.message_count),
            environment_type=self.build_environment_type(observation.environment_type),
            audio_risk_flag=bool(observation.audio_risk_flag),
            video_risk_flag=bool(observation.video_risk_flag),
            rule_hit_level=self.build_rule_hit_level(observation.rule_hit_level),
        )


state_builder = StateBuilder()
