from sqlalchemy.ext.asyncio import AsyncSession

from app.core.defense_levels import get_risk_level
from app.models.user import User
from app.services.email_service import email_service

from .mdp_types import DefenseAction, MDPObservation, PolicyDecision, RiskLevel, RuleHitLevel
from .policy_repository import PolicyRepository
from .state_builder import state_builder


class DynamicDefenseAgent:
    def __init__(self, q_table_path: str = "app/services/mdp_defense/q_table_policy.json"):
        self.policy_repository = PolicyRepository(q_table_path)
        self.policy, self.policy_version = self.policy_repository.load_policy()

    def reload_policy(self):
        self.policy, self.policy_version = self.policy_repository.load_policy()

    def select_action(self, user: User | None, observation: MDPObservation) -> PolicyDecision:
        state = state_builder.build_state(user, observation)

        if state.policy_key() in self.policy:
            return PolicyDecision(
                state=state,
                action=DefenseAction(self.policy[state.policy_key()]),
                policy_version=self.policy_version,
                reason_codes=["policy_exact_match"],
                fallback_used=False,
            )

        if state.legacy_policy_key() in self.policy:
            return PolicyDecision(
                state=state,
                action=DefenseAction(self.policy[state.legacy_policy_key()]),
                policy_version=self.policy_version,
                reason_codes=["legacy_policy_match"],
                fallback_used=False,
            )

        return self._fallback_decision(state)

    def _fallback_decision(self, state) -> PolicyDecision:
        reason_codes = ["fallback_rule_based"]

        if state.rule_hit_level in [RuleHitLevel.HIGH, RuleHitLevel.CRITICAL]:
            return PolicyDecision(
                state=state,
                action=DefenseAction.LEVEL_3,
                policy_version="fallback-rule-v1",
                reason_codes=reason_codes + ["high_rule_hit"],
                fallback_used=True,
            )

        if state.risk_level == RiskLevel.HIGH:
            return PolicyDecision(
                state=state,
                action=DefenseAction.LEVEL_3,
                policy_version="fallback-rule-v1",
                reason_codes=reason_codes + ["high_risk_bucket"],
                fallback_used=True,
            )

        if state.risk_level == RiskLevel.MEDIUM:
            if (
                state.rule_hit_level == RuleHitLevel.MEDIUM
                or state.audio_risk_flag
                or state.video_risk_flag
                or state.interaction_depth.value == "DEEP"
            ):
                return PolicyDecision(
                    state=state,
                    action=DefenseAction.LEVEL_2,
                    policy_version="fallback-rule-v1",
                    reason_codes=reason_codes + ["medium_risk_with_signal"],
                    fallback_used=True,
                )

            return PolicyDecision(
                state=state,
                action=DefenseAction.LEVEL_2,
                policy_version="fallback-rule-v1",
                reason_codes=reason_codes + ["medium_risk_default"],
                fallback_used=True,
            )

        return PolicyDecision(
            state=state,
            action=DefenseAction.LEVEL_1,
            policy_version="fallback-rule-v1",
            reason_codes=reason_codes + ["low_risk_default"],
            fallback_used=True,
        )

    async def notify_guardian_by_email(
        self,
        db: AsyncSession,
        user_id: int,
        risk_level: str,
        details: str = "",
    ) -> bool:
        success_count = await email_service.send_guardian_alert_by_family(
            db=db,
            user_id=user_id,
            risk_level=risk_level,
            details=details,
        )
        return success_count > 0

    def get_defense_action(
        self,
        user: User | None,
        current_risk_score: float,
        message_count: int,
        environment_type: str = "unknown",
        audio_risk_flag: bool = False,
        video_risk_flag: bool = False,
        rule_hit_level: int = 0,
        text_conf: float = 0.0,
        audio_conf: float = 0.0,
        video_conf: float = 0.0,
    ) -> PolicyDecision:
        observation = MDPObservation(
            user_id=getattr(user, "user_id", 0) or 0,
            call_id=0,
            risk_score=current_risk_score,
            message_count=message_count,
            environment_type=environment_type,
            audio_risk_flag=audio_risk_flag,
            video_risk_flag=video_risk_flag,
            rule_hit_level=rule_hit_level,
            text_conf=text_conf,
            audio_conf=audio_conf,
            video_conf=video_conf,
        )
        return self.select_action(user, observation)

    async def get_defense_action_with_notification(
        self,
        db: AsyncSession,
        user_id: int,
        user: User | None,
        current_risk_score: float,
        message_count: int,
        details: str = "",
        environment_type: str = "unknown",
        audio_risk_flag: bool = False,
        video_risk_flag: bool = False,
        rule_hit_level: int = 0,
    ) -> PolicyDecision:
        decision = self.get_defense_action(
            user=user,
            current_risk_score=current_risk_score,
            message_count=message_count,
            environment_type=environment_type,
            audio_risk_flag=audio_risk_flag,
            video_risk_flag=video_risk_flag,
            rule_hit_level=rule_hit_level,
        )

        if decision.action == DefenseAction.LEVEL_3:
            mapped_risk_level = get_risk_level(decision.action.value)
            await self.notify_guardian_by_email(
                db=db,
                user_id=user_id,
                risk_level=mapped_risk_level,
                details=details,
            )

        return decision