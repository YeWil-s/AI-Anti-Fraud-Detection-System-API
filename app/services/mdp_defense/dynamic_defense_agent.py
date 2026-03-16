from app.models.user import User
from .mdp_env import UserVulnerability, RiskLevel, InteractionDepth, DefenseAction
import json
import os

def calculate_vulnerability(user: User) -> UserVulnerability:
    """
    将现有的用户画像维度（角色、职业、婚姻等）映射为易感度等级。
    这里契合了大赛背景中提到的不同人群被骗倾向：
    老年人/儿童/学生 -> 容易受骗；稳定职业且已婚 -> 防范较强
    """
    # 1. 安全获取字段（防止数据库中存在 None 导致 .lower() 报错）
    profession = (getattr(user, 'profession', '') or '').lower()
    role_type = getattr(user, 'role_type', '') or ''
    marital_status = getattr(user, 'marital_status', '') or ''
    
    # 2. 判断高危群体：结合 role_type (老人/儿童/学生) 和 profession (退休/无业)
    high_risk_roles = ['老人', '儿童', '学生']
    high_risk_occupations = ['学生', '退休', '无业']
    
    # 如果角色属于高危，或者职业属于高危，直接判定为高易感度
    if any(role in role_type for role in high_risk_roles) or \
       any(job in profession for job in high_risk_occupations):
        return UserVulnerability.HIGH
        
    # 3. 判断低危群体：防范意识较强的职业 + 相对稳定的生活状态(已婚)
    low_risk_occupations = ['程序员', '教师', '公务员', '医生', 'it', 'teacher']
    if any(job in profession for job in low_risk_occupations) and marital_status == '已婚':
        return UserVulnerability.LOW
        
    # 4. 其他普通情况默认为中易感度
    return UserVulnerability.MEDIUM

class DynamicDefenseAgent:
    def __init__(self, q_table_path="app/services/mdp_defense/q_table_policy.json"):
        """初始化 Agent，加载训练好的策略表"""
        self.q_table = {}
        if os.path.exists(q_table_path):
            with open(q_table_path, 'r', encoding='utf-8') as f:
                self.q_table = json.load(f)
        else:
            print(f"Warning: Q-Table 文件未找到: {q_table_path}")

    def get_best_action(self, vuln: UserVulnerability, risk: RiskLevel, depth: InteractionDepth) -> DefenseAction:
        """根据当前状态查询最佳防御动作"""
        # 这里需要实现你的查表逻辑
        # 比如将状态转为 string key 去 q_table 里面查
        state_key = f"{vuln.value}_{risk.value}_{depth.value}"
        
        if state_key in self.q_table:
            action_value = self.q_table[state_key]
            return DefenseAction(action_value)
            
        # 如果没查到，默认返回低级别防御
        return DefenseAction.LEVEL_1