from __future__ import annotations

from app.models.call_record import DetectionResult

from .mdp_types import DefenseAction, InteractionDepth, MDPState, RiskLevel, RuleHitLevel, UserVulnerability


class RewardBuilder:

    def calculate_step_reward(
        self,
        current_state: MDPState,
        action: DefenseAction,
        next_state: MDPState | None = None,
        final_result: DetectionResult | str | None = None,
    ) -> float:
        reward = self._action_alignment_reward(current_state, action)

        if next_state:
            reward += self._transition_reward(current_state, next_state)

        if final_result:
            reward += self.calculate_final_outcome_reward(action, final_result)

        return round(reward, 2)

    def _action_alignment_reward(self, state: MDPState, action: DefenseAction) -> float:
        severe_signal = (
            state.risk_level == RiskLevel.HIGH
            or state.rule_hit_level in [RuleHitLevel.HIGH, RuleHitLevel.CRITICAL]
            or (state.audio_risk_flag and state.video_risk_flag)
        )
        elevated_signal = (
            state.risk_level == RiskLevel.MEDIUM
            or state.rule_hit_level in [RuleHitLevel.MEDIUM, RuleHitLevel.LOW]
            or state.audio_risk_flag
            or state.video_risk_flag
        )

        if severe_signal:
            return {
                DefenseAction.LEVEL_1: -120.0,
                DefenseAction.LEVEL_2: 20.0,
                DefenseAction.LEVEL_3: 90.0,
            }[action]

        if elevated_signal:
            if state.user_vulnerability == UserVulnerability.HIGH and state.interaction_depth.value == "DEEP":
                return {
                    DefenseAction.LEVEL_1: -60.0,
                    DefenseAction.LEVEL_2: 45.0,
                    DefenseAction.LEVEL_3: 60.0,
                }[action]
            return {
                DefenseAction.LEVEL_1: -20.0,
                DefenseAction.LEVEL_2: 50.0,
                DefenseAction.LEVEL_3: 10.0,
            }[action]

        return {
            DefenseAction.LEVEL_1: 20.0,
            DefenseAction.LEVEL_2: -25.0,
            DefenseAction.LEVEL_3: -80.0,
        }[action]

    def _transition_reward(self, current_state: MDPState, next_state: MDPState) -> float:
        risk_delta = self._risk_score(next_state.risk_level) - self._risk_score(current_state.risk_level)
        if risk_delta < 0:
            return 25.0
        if risk_delta > 0:
            return -25.0

        if next_state.interaction_depth != current_state.interaction_depth:
            if self._depth_score(next_state.interaction_depth) > self._depth_score(current_state.interaction_depth):
                return -5.0
            return 5.0
        return 0.0

    def calculate_final_outcome_reward(self, action: DefenseAction, final_result: DetectionResult | str) -> float:
        final_value = final_result.value if isinstance(final_result, DetectionResult) else str(final_result)
        if final_value == DetectionResult.FAKE.value:
            return {
                DefenseAction.LEVEL_1: -80.0,
                DefenseAction.LEVEL_2: 25.0,
                DefenseAction.LEVEL_3: 60.0,
            }[action]
        if final_value == DetectionResult.SUSPICIOUS.value:
            return {
                DefenseAction.LEVEL_1: -20.0,
                DefenseAction.LEVEL_2: 20.0,
                DefenseAction.LEVEL_3: 10.0,
            }[action]
        return {
            DefenseAction.LEVEL_1: 10.0,
            DefenseAction.LEVEL_2: -20.0,
            DefenseAction.LEVEL_3: -50.0,
        }[action]

    def _risk_score(self, risk_level: RiskLevel) -> int:
        mapping = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
        }
        return mapping[risk_level]

    def _depth_score(self, depth: InteractionDepth) -> int:
        mapping = {
            InteractionDepth.SHALLOW: 0,
            InteractionDepth.MEDIUM: 1,
            InteractionDepth.DEEP: 2,
        }
        return mapping[depth]


reward_builder = RewardBuilder()
