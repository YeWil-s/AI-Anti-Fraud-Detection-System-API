# app/services/mdp_defense/mdp_env.py
from enum import Enum
from typing import Tuple

# 1. 定义状态空间 
class UserVulnerability(Enum):
    LOW = 0     # 低易感人群（防范意识强）
    MEDIUM = 1  # 中易感人群
    HIGH = 2    # 高易感人群（如老年人、未成年人，易受诱导）

class RiskLevel(Enum):
    LOW = 0     # 低危 (LLM打分 0-40)
    MEDIUM = 1  # 中危 (LLM打分 41-80)
    HIGH = 2    # 高危 (LLM打分 81-100)

class InteractionDepth(Enum):
    SHALLOW = 0 # 浅度交互 (1-3轮)
    MEDIUM = 1  # 中度交互 (4-10轮)
    DEEP = 2    # 深度交互 (>10轮，被洗脑风险极高)

# 2. 定义动作空间 (Action Space)
class DefenseAction(Enum):
    LEVEL_1 = 0  # 一级防御：隐式监测、侧边温和提示
    LEVEL_2 = 1  # 二级防御：强力干预、多模态核验、推送案例
    LEVEL_3 = 2  # 三级防御：强制阻断、监护人联动告警

# 3. 定义 MDP 环境类
class AntiFraudMDPEnv:
    def __init__(self):
        # 状态空间大小：3 * 3 * 3 = 27 种状态
        self.state_space_size = len(UserVulnerability) * len(RiskLevel) * len(InteractionDepth)
        self.action_space_size = len(DefenseAction)

    def get_state_index(self, vuln: UserVulnerability, risk: RiskLevel, depth: InteractionDepth) -> int:
        """将多维状态映射为一维索引，方便后续 Q-Table 查表"""
        return vuln.value * 9 + risk.value * 3 + depth.value

    def calculate_reward(self, vuln: UserVulnerability, risk: RiskLevel, depth: InteractionDepth, action: DefenseAction) -> float:
        """
        升级版奖励函数：融入画像和深度，强制模型学会“因人而异”的动态阈值
        """
        reward = 0.0

        # 1. 绝对低危：尽量不打扰
        if risk == RiskLevel.LOW:
            if action == DefenseAction.LEVEL_3:
                return -200.0  # 绝对惩罚误报
            elif action == DefenseAction.LEVEL_2:
                return -50.0
            else:
                return 10.0    # 正常只给弱提示（Level 1）

        # 2. 绝对高危：绝不姑息，直接最高防御
        elif risk == RiskLevel.HIGH:
            if action == DefenseAction.LEVEL_1:
                return -500.0  # 致命惩罚：漏报
            elif action == DefenseAction.LEVEL_2:
                return -100.0
            else:
                return 100.0   # 奖励监护人联动（Level 3）

        # 3. 中危地带：核心！这里要拉开差距，体现动态画像！
        else: # RiskLevel.MEDIUM
            if vuln == UserVulnerability.HIGH:
                # 【场景 A：高易感人群（如老人、未成年）】
                # 他们在“中危”也很容易受骗，且如果是深交互，极其危险。
                if depth == InteractionDepth.DEEP:
                    if action == DefenseAction.LEVEL_3:
                        reward = 100.0  # 鼓励直接联动监护人！(动态阈值起效)
                    else:
                        reward = -100.0 # 惩罚干预不够
                else:
                    if action == DefenseAction.LEVEL_2:
                        reward = 50.0
                    else:
                        reward = -50.0
            
            elif vuln == UserVulnerability.LOW:
                # 【场景 B：低易感人群（防范意识极强的青壮年）】
                # 尽量少打扰他们，以免引发反感。
                if depth == InteractionDepth.SHALLOW:
                    if action == DefenseAction.LEVEL_1:
                        reward = 50.0   # 刚开始聊中危，给个小提示就行 (不打扰)
                    else:
                        reward = -80.0  # 惩罚过度弹窗
                else:
                    if action == DefenseAction.LEVEL_2:
                        reward = 50.0
                    else:
                        reward = -20.0
            
            else:
                # 【场景 C：普通人群】中规中矩
                if action == DefenseAction.LEVEL_2:
                    reward = 50.0
                else:
                    reward = -20.0

        return reward