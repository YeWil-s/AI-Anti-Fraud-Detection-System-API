from .mdp_types import DefenseAction, InteractionDepth, MDPState, RiskLevel, UserVulnerability
from .reward_builder import reward_builder


class AntiFraudMDPEnv:
    """
    真实闭环下的 MDP 环境定义。

    该环境不再假定“理想化”的状态转移，而是复用线上状态模型和奖励规则，
    让离线训练直接消费真实轨迹事件。
    """

    def __init__(self):
        self.action_space_size = len(DefenseAction)

    def get_state_index(self, state: MDPState) -> str:
        """在线/离线统一使用 state_key 作为状态标识。"""
        return state.policy_key()

    def calculate_reward(
        self,
        state: MDPState,
        action: DefenseAction,
        next_state: MDPState | None = None,
        final_result: str | None = None,
    ) -> float:
        return reward_builder.calculate_step_reward(
            current_state=state,
            action=action,
            next_state=next_state,
            final_result=final_result,
        )

    def get_legacy_state_index(
        self,
        vuln: UserVulnerability,
        risk: RiskLevel,
        depth: InteractionDepth,
    ) -> str:
        """兼容旧版三维状态 key。"""
        return f"{vuln.value}_{risk.value}_{depth.value}"