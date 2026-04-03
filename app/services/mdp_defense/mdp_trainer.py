import asyncio
from collections import defaultdict
from typing import Dict, List

from app.db.database import AsyncSessionLocal

from .mdp_env import AntiFraudMDPEnv
from .mdp_types import DefenseAction
from .offline_dataset import TrajectorySample, offline_dataset
from .policy_repository import PolicyRepository


class MDPTrainer:
    def __init__(self, env: AntiFraudMDPEnv, alpha: float = 0.1, gamma: float = 0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def train_from_samples(self, samples: List[TrajectorySample]):
        for sample in samples:
            state_key = sample.state.policy_key()
            next_state_key = sample.next_state.policy_key()
            action_idx = sample.action.value

            best_next_q = max(self.q_table[next_state_key].values(), default=0.0)
            td_target = sample.reward + self.gamma * best_next_q
            current_q = self.q_table[state_key][action_idx]
            self.q_table[state_key][action_idx] = current_q + self.alpha * (td_target - current_q)

    def export_policy(self, repository: PolicyRepository) -> str:
        policy = {}
        for state_key, action_values in self.q_table.items():
            if not action_values:
                continue
            best_action = max(action_values.items(), key=lambda item: item[1])[0]
            policy[state_key] = int(best_action)
        return repository.save_policy(
            policy,
            metadata={
                "trainer": "offline_q_learning",
                "state_schema_version": "v2",
                "sample_count": sum(len(v) for v in self.q_table.values()),
            },
        )


async def train_policy_from_database():
    env = AntiFraudMDPEnv()
    trainer = MDPTrainer(env)
    repository = PolicyRepository()

    async with AsyncSessionLocal() as db:
        samples = await offline_dataset.load_completed_samples(db)

    if not samples:
        print(
            "未加载到任何已完成轨迹（需 mdp_decision_events 中 next_state_json、reward 均已填写）。"
            "跳过导出，保留现有 q_table_policy.json，避免覆盖为冷启动空策略。"
        )
        return

    trainer.train_from_samples(samples)
    version = trainer.export_policy(repository)
    print(f"策略训练完成，已导出版本: {version}")


if __name__ == "__main__":
    asyncio.run(train_policy_from_database())