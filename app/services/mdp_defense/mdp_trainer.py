# app/services/mdp_defense/mdp_trainer.py
import numpy as np
import json
import random
import os
from .mdp_env import AntiFraudMDPEnv, UserVulnerability, RiskLevel, InteractionDepth, DefenseAction

class MDPTrainer:
    def __init__(self, env: AntiFraudMDPEnv, seed: int = 42):
        self.env = env
        # 初始化 Q-Table，大小为 (27, 3)，全部填 0
        self.q_table = np.zeros((self.env.state_space_size, self.env.action_space_size))
        
        # Q-Learning 超参数
        self.alpha = 0.1       # 学习率：新知识覆盖老知识的比例
        self.gamma = 0.9       # 折扣因子：对未来长远收益的重视程度
        self.epsilon = 1.0     # 探索率：初始100%随机探索
        self.epsilon_decay = 0.995 # 探索衰减：慢慢减少随机，开始利用已有经验
        self.epsilon_min = 0.01
        self.episodes = 20000  # 训练轮数（模拟 2 万次诈骗对抗）
        
        # 设置随机种子，确保结果可复现
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def simulate_transition(self, vuln: UserVulnerability, risk: RiskLevel, depth: InteractionDepth, action: DefenseAction):
        """
        模拟状态转移：采取防御动作后，诈骗风险会如何变化？
        这里体现了“不同画像人群的不同反应”
        """
        next_risk = risk
        next_depth = InteractionDepth(min(2, depth.value + 1)) # 默认交互加深

        if action == DefenseAction.LEVEL_3:
            # 三级防御：直接阻断并联动监护人。风险立刻降为 LOW，交互终止
            next_risk = RiskLevel.LOW
            next_depth = InteractionDepth.SHALLOW
            
        elif action == DefenseAction.LEVEL_2:
            # 二级防御：强力干预
            if vuln == UserVulnerability.LOW:
                next_risk = RiskLevel.LOW # 防范意识强的人被强力干预后立刻清醒
            elif vuln == UserVulnerability.HIGH and random.random() < 0.4:
                # 易感人群有 40% 概率无视二级防御，继续被骗，风险升高
                next_risk = RiskLevel(min(2, risk.value + 1))
            else:
                next_risk = RiskLevel(max(0, risk.value - 1)) # 风险降低
                
        elif action == DefenseAction.LEVEL_1:
            # 一级防御：温和提示
            if vuln == UserVulnerability.HIGH:
                # 易感人群极大概率无视温和提示，风险恶化
                if random.random() < 0.8:
                    next_risk = RiskLevel(min(2, risk.value + 1))
            elif vuln == UserVulnerability.LOW and random.random() < 0.8:
                # 防范意识强的人看到提示就停止了
                next_risk = RiskLevel.LOW

        return vuln, next_risk, next_depth

    def train(self):
        print("开始离线训练 MDP 策略...")
        for episode in range(self.episodes):
            # 随机初始化一个初始状态
            vuln = random.choice(list(UserVulnerability))
            risk = random.choice(list(RiskLevel))
            depth = random.choice(list(InteractionDepth))
            
            # 模拟 5 步交互（一个诈骗周期）
            for step in range(5):
                state_idx = self.env.get_state_index(vuln, risk, depth)
                
                # Epsilon-Greedy 策略选择动作
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(list(DefenseAction)) # 随机探索
                else:
                    action_idx = np.argmax(self.q_table[state_idx]) # 利用已有经验
                    action = DefenseAction(action_idx)
                
                # 获取奖励
                reward = self.env.calculate_reward(vuln, risk, depth, action)
                
                # 转移到下一个状态
                next_vuln, next_risk, next_depth = self.simulate_transition(vuln, risk, depth, action)
                next_state_idx = self.env.get_state_index(next_vuln, next_risk, next_depth)
                
                # 更新 Q-Table核心公式
                best_next_action = np.argmax(self.q_table[next_state_idx])
                td_target = reward + self.gamma * self.q_table[next_state_idx][best_next_action]
                self.q_table[state_idx][action.value] += self.alpha * (td_target - self.q_table[state_idx][action.value])
                
                # 状态推进
                vuln, risk, depth = next_vuln, next_risk, next_depth
                
                # 如果风险降为低，提前结束本轮
                if risk == RiskLevel.LOW and action == DefenseAction.LEVEL_3:
                    break
            
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        print("训练完成！")

    def export_policy(self, filename="q_table_policy.json"):
        """导出最终的最优策略为 JSON 格式"""
        policy = {}
        for v in UserVulnerability:
            for r in RiskLevel:
                for d in InteractionDepth:
                    state_idx = self.env.get_state_index(v, r, d)
                    best_action_idx = int(np.argmax(self.q_table[state_idx]))
                    
                    # 构造易读的 Key，如 "HIGH_HIGH_DEEP"
                    state_key = f"{v.name}_{r.name}_{d.name}"
                    policy[state_key] = best_action_idx
                    
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(policy, f, indent=4)
        print(f"策略表已导出至: {filename}")

if __name__ == "__main__":
    env = AntiFraudMDPEnv()
    trainer = MDPTrainer(env)
    trainer.train()
    # 将策略文件保存在同一目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trainer.export_policy(os.path.join(current_dir, "q_table_policy.json"))